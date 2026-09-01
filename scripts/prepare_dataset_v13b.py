#!/usr/bin/env python3
"""Build the leakage-gated DATASET_V13B pilot without touching V13 history.

V13B is intentionally conservative: only explicit APPROVED sources enter its
production splits, every direct source is class-balanced using content pairs,
and virtual partial/mixed examples are created only from TRAIN-owned bases.
The builder records unmet gates instead of pretending the dataset is ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import load_audio

OUTPUT = ROOT / "data/splits_v13b"
EXTERNAL_DIRS = ("raw", "generated", "processed", "paired", "partial", "mixed",
                 "channel_augmented", "manifests", "splits", "provenance", "licenses",
                 "frozen_versions", "training_runs")
SEED = 23674913
RATIOS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def resolve(value: object) -> pathlib.Path:
    path = pathlib.Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def canonicalize(row: pd.Series, data_root: pathlib.Path) -> pd.Series:
    """Apply one label-independent file/channel policy to every direct row."""
    source = resolve(row.path)
    original_sha = sha256(source)
    destination = data_root / "processed/canonical16k_pcm16" / str(row.source) / (
        f"{original_sha}.wav")
    if not destination.is_file():
        wave, _ = load_audio(str(source), target_sr=16_000)
        wave = np.asarray(wave, dtype=np.float32)
        if not len(wave):
            raise RuntimeError(f"empty audio: {source}")
        # The same deterministic peak ceiling is applied to both labels.  It
        # avoids codec/source loudness becoming a free label without erasing
        # within-file forensic structure.
        peak = float(np.max(np.abs(wave)))
        if peak > 0.95:
            wave = wave * (0.95 / peak)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, wave, 16_000, subtype="PCM_16")
    result = row.copy()
    result["source_path"] = str(source)
    result["source_audio_sha256"] = original_sha
    result["path"] = str(destination)
    result["audio_sha256"] = sha256(destination)
    result["sample_rate"] = 16_000
    result["codec"] = "PCM_S16LE"
    result["extension"] = ".wav"
    result["channel_policy"] = "canonical16k_pcm16_label_independent_v1"
    result["augment"] = "none"
    return result


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "content_group": "original_id", "generator_family": "generator",
        "speaker_id": "source", "near_duplicate_group": "split_group_id",
    }
    for target, fallback in aliases.items():
        if target not in result:
            result[target] = result[fallback]
        result[target] = result[target].fillna(result[fallback]).astype(str)
    if "competition_use_status" not in result:
        result["competition_use_status"] = "REVIEW_REQUIRED"
    result["data_role"] = "paired_core"
    result["base_audio_id"] = result["content_group"]
    result["base_voice_id"] = np.where(
        result.voice_present.eq(1), result.content_group, "ABSENT")
    result["base_music_id"] = np.where(
        result.music_present.eq(1), result.content_group, "ABSENT")
    result["voice_content_group"] = result["base_voice_id"]
    result["music_content_group"] = result["base_music_id"]
    result["voice_generator_family"] = np.where(
        result.voice_present.eq(1), result.generator_family, "ABSENT")
    result["music_generator_family"] = np.where(
        result.music_present.eq(1), result.generator_family, "ABSENT")
    result["parent_real_id"] = np.where(
        result.file_fake.eq(0), result.content_group, "ABSENT")
    result["parent_fake_id"] = np.where(
        result.file_fake.eq(1), result.content_group, "ABSENT")
    result["augmentation"] = "none"
    return result


def apply_explicit_approval(frame: pd.DataFrame, source: str, registry: dict) -> pd.DataFrame:
    entry = registry["sources"].get(source, {})
    required = ("status", "approval_basis", "license_source",
                "license_snapshot_sha256", "reviewed_at")
    missing = [name for name in required if not str(entry.get(name, "")).strip()]
    if entry.get("status") != "APPROVED" or missing:
        raise RuntimeError(f"{source}: explicit approval evidence incomplete: {missing}")
    result = frame.copy()
    if "source_url" not in result or result.source_url.fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{source}: row source_url is required for approval")
    if "license" not in result or result.license.fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError(f"{source}: row license is required for approval")
    result["competition_use_status"] = "APPROVED"
    result["approval_basis"] = entry["approval_basis"]
    result["license_source"] = entry["license_source"]
    result["license_snapshot_sha256"] = entry["license_snapshot_sha256"]
    result["reviewed_at"] = entry["reviewed_at"]
    return result


def explicit_generator_roles(source: str, frame: pd.DataFrame, split_config: dict) -> dict[str, str]:
    if source not in split_config.get("sources", {}):
        raise RuntimeError(f"generator split config missing source: {source}")
    config = split_config["sources"][source]
    roles: dict[str, str] = {}
    for key, role in (("train", "train"), ("generator_val", "val_generator_disjoint"),
                      ("cal", "cal_v13b")):
        for family in config.get(key, []):
            if family in roles:
                raise RuntimeError(f"generator appears in multiple roles: {family}")
            roles[family] = role
    observed = set(frame.loc[frame.file_fake.eq(1), "generator_family"].astype(str))
    unknown = sorted(observed - set(roles))
    if unknown:
        raise RuntimeError(f"unknown generators require explicit review: {unknown}")
    return roles


def paired_by_generator(frame: pd.DataFrame, source: str, split_config: dict) -> pd.DataFrame:
    fake = frame[frame.file_fake.eq(1)].copy()
    family_role = explicit_generator_roles(source, frame, split_config)
    roles = []
    real_by_group = {group: values.iloc[0] for group, values in
                     frame[frame.file_fake.eq(0)].groupby("content_group")}
    for _, fake_row in fake.iterrows():
        group = str(fake_row.content_group)
        if group not in real_by_group:
            continue
        role = family_role[str(fake_row.generator_family)]
        real_row = real_by_group[group].copy()
        fake_copy = fake_row.copy()
        real_row["v13b_role"], fake_copy["v13b_role"] = role, role
        roles.extend((real_row, fake_copy))
    return pd.DataFrame(roles).drop_duplicates(
        ["v13b_role", "content_group", "file_fake", "generator_family"])


def echoes_pairs(frame: pd.DataFrame, split_config: dict) -> pd.DataFrame:
    fake = frame[frame.file_fake.eq(1)].copy()
    family_role = explicit_generator_roles("echoes_fma_paired", frame, split_config)
    rows = []
    for group, values in frame.groupby("content_group"):
        real = values[values.file_fake.eq(0)]
        candidates = values[values.file_fake.eq(1)].copy()
        if real.empty or candidates.empty:
            continue
        candidates["stable"] = candidates.generator_family.map(
            lambda family: hashlib.sha256(f"{group}::{family}".encode()).hexdigest())
        selected = candidates.sort_values("stable").iloc[0].drop(labels="stable")
        role = family_role[str(selected.generator_family)]
        real_row, fake_row = real.iloc[0].copy(), selected.copy()
        real_row["v13b_role"], fake_row["v13b_role"] = role, role
        rows.extend((real_row, fake_row))
    return pd.DataFrame(rows)


def partial_rows(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component in ("voice", "music"):
        present = f"{component}_present"
        fake_label = f"{component}_fake"
        real = train[train[present].eq(1) & train[fake_label].eq(0) &
                     train.data_role.eq("paired_core")].reset_index(drop=True)
        fake = train[train[present].eq(1) & train[fake_label].eq(1) &
                     train.data_role.eq("paired_core")].reset_index(drop=True)
        if len(real) < 2 or fake.empty:
            continue
        for index, ratio in enumerate(RATIOS):
            for replicate in range(2):
                real_a = real.iloc[(index * 2 + replicate) % len(real)]
                real_b = real.iloc[(index * 2 + replicate + 1) % len(real)]
                fake_row = fake.iloc[(index * 2 + replicate) % len(fake)]
                for label, second in ((1, fake_row), (0, real_b)):
                    row = real_a.copy()
                    row["path"] = f"PARTIAL::{component}::{real_a.path}|{second.path}"
                    row["file_fake"] = label
                    row["voice_fake"] = label if component == "voice" else 0
                    row["music_fake"] = label if component == "music" else 0
                    row["partial_fake_ratio"] = ratio
                    row["partial_fake_position"] = ("middle", "start", "end")[index % 3]
                    row["partial_crossfade_sec"] = 0.02
                    row["generator_family"] = (f"partial::{second.generator_family}"
                                               if label else "REAL_CONTROL")
                    row["generator"] = row["generator_family"]
                    row["source"] = f"v13b_partial_{component}"
                    row["dataset"] = f"v13b_partial_{component}"
                    row["data_role"] = "partial_fake" if label else "partial_real_control"
                    row["content_group"] = f"partial::{component}::{real_a.content_group}::{second.content_group}"
                    row["split_group_id"] = row["content_group"]
                    row["base_audio_id"] = f"{real_a.content_group}|{second.content_group}"
                    row["base_voice_id"] = (row["base_audio_id"] if component == "voice"
                                            else "ABSENT")
                    row["base_music_id"] = (row["base_audio_id"] if component == "music"
                                            else "ABSENT")
                    row["voice_content_group"] = row["base_voice_id"]
                    row["music_content_group"] = row["base_music_id"]
                    row["voice_generator_family"] = (
                        str(second.generator_family) if component == "voice" and label else
                        "REAL_CONTROL" if component == "voice" else "ABSENT")
                    row["music_generator_family"] = (
                        str(second.generator_family) if component == "music" and label else
                        "REAL_CONTROL" if component == "music" else "ABSENT")
                    row["parent_real_id"] = str(real_a.content_group)
                    row["parent_fake_id"] = (str(second.content_group) if label else "ABSENT")
                    row["audio_sha256"] = "VIRTUAL"
                    rows.append(row)
    return pd.DataFrame(rows)


def mixed_rows(train: pd.DataFrame, per_state: int = 24) -> pd.DataFrame:
    core = train[train.data_role.eq("paired_core")]
    voice = {label: core[core.voice_present.eq(1) & core.voice_fake.eq(label)].reset_index(drop=True)
             for label in (0, 1)}
    music = {label: core[core.music_present.eq(1) & core.music_fake.eq(label)].reset_index(drop=True)
             for label in (0, 1)}
    states = {"RR": (0, 0), "RF": (0, 1), "FR": (1, 0), "FF": (1, 1)}
    rows = []
    modes = ("simultaneous", "voice_then_music", "music_then_voice",
             "partial_overlap", "crossfade")
    for state, (voice_fake, music_fake) in states.items():
        for index in range(per_state):
            v = voice[voice_fake].iloc[index % len(voice[voice_fake])]
            m = music[music_fake].iloc[(index * 7 + 3) % len(music[music_fake])]
            row = v.copy()
            row["path"] = f"MIX::{v.path}|{m.path}"
            row["file_fake"] = max(voice_fake, music_fake)
            row["voice_fake"], row["music_fake"] = voice_fake, music_fake
            row["voice_present"], row["music_present"] = 1, 1
            row["mix_state"] = state
            row["mix_mode"] = modes[index % len(modes)]
            row["mix_snr_db"] = (-8.0, -3.0, 0.0, 3.0, 8.0)[index % 5]
            row["mix_crossfade_sec"] = 0.25
            row["mix_overlap_fraction"] = (0.25, 0.5, 0.75)[index % 3]
            row["mix_gap_sec"] = (0.0, 0.2)[index % 2]
            row["mix_voice_gain_db"] = (-3.0, 0.0, 3.0)[index % 3]
            row["mix_music_gain_db"] = (3.0, 0.0, -3.0)[index % 3]
            row["source"] = row["dataset"] = "v13b_mixed_balanced"
            row["data_role"] = "mixed"
            row["generator_family"] = f"MIX::{v.generator_family}+{m.generator_family}"
            row["generator"] = row["generator_family"]
            row["content_group"] = f"mix::{v.content_group}::{m.content_group}::{index}"
            row["split_group_id"] = row["content_group"]
            row["base_audio_id"] = f"{v.content_group}|{m.content_group}"
            row["base_voice_id"] = str(v.content_group)
            row["base_music_id"] = str(m.content_group)
            row["voice_content_group"] = str(v.content_group)
            row["music_content_group"] = str(m.content_group)
            row["voice_generator_family"] = str(v.generator_family)
            row["music_generator_family"] = str(m.generator_family)
            row["parent_real_id"] = "|".join(
                str(item.content_group) for item, fake in ((v, voice_fake), (m, music_fake))
                if not fake) or "ABSENT"
            row["parent_fake_id"] = "|".join(
                str(item.content_group) for item, fake in ((v, voice_fake), (m, music_fake))
                if fake) or "ABSENT"
            row["audio_sha256"] = "VIRTUAL"
            rows.append(row)
    return pd.DataFrame(rows)


def set_overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    def identifiers(frame: pd.DataFrame) -> set[str]:
        values: set[str] = set()
        for raw in frame[column].dropna().astype(str):
            values.update(part for part in raw.split("|") if part not in {
                "", "nan", "VIRTUAL", "ABSENT", "REAL_CONTROL"})
        return values

    a = identifiers(left)
    b = identifiers(right)
    return len(a & b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.getenv(
        "AI_VOICE_DATA_ROOT", str(pathlib.Path.home() / "ai_voice_data_v13b")))
    parser.add_argument("--skip-canonicalize", action="store_true")
    args = parser.parse_args()
    data_root = pathlib.Path(args.data_root).resolve()
    for directory in EXTERNAL_DIRS:
        (data_root / directory).mkdir(parents=True, exist_ok=True)

    registry = json.loads((ROOT / "configs/v13b/source_registry.json").read_text(encoding="utf-8"))
    split_config = json.loads(
        (ROOT / "configs/v13b/generator_split.json").read_text(encoding="utf-8"))
    public = enrich(pd.read_csv(ROOT / "data/splits_v9_candidate/manifest.csv"))
    mlaad = apply_explicit_approval(
        public[public.source.eq("mlaad_tiny_matched")], "mlaad_tiny_matched", registry)
    dfadd_path = data_root / "manifests/dfadd_paired_v13b.csv"
    if not dfadd_path.is_file():
        raise FileNotFoundError(
            f"run scripts/acquire_dfadd_pairs_v13b.py first: {dfadd_path}")
    dfadd = apply_explicit_approval(
        enrich(pd.read_csv(dfadd_path)), "dfadd_vctk_paired", registry)
    echoes = apply_explicit_approval(
        public[public.source.eq("echoes_fma_paired")], "echoes_fma_paired", registry)

    selected = pd.concat([
        paired_by_generator(mlaad, "mlaad_tiny_matched", split_config),
        paired_by_generator(dfadd, "dfadd_vctk_paired", split_config),
        echoes_pairs(echoes, split_config),
    ], ignore_index=True, sort=False)
    if not selected.competition_use_status.eq("APPROVED").all():
        raise RuntimeError("V13B production core contains non-APPROVED rows")
    if not args.skip_canonicalize:
        selected = pd.DataFrame([canonicalize(row, data_root) for _, row in selected.iterrows()])

    direct = selected.copy()
    train_core = direct[direct.v13b_role.eq("train")].copy()
    partial = partial_rows(train_core)
    mixed = mixed_rows(train_core)
    train = pd.concat([train_core, partial, mixed], ignore_index=True, sort=False)
    val = direct[direct.v13b_role.eq("val_generator_disjoint")].copy()
    cal = direct[direct.v13b_role.eq("cal_v13b")].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {
        "manifest": pd.concat([direct, partial, mixed], ignore_index=True, sort=False),
        "train": train,
        "val_generator_disjoint": val,
        "cal_v13b": cal,
        "partial_train": partial,
        "mixed_train": mixed,
    }
    file_meta = {}
    for name, frame in frames.items():
        path = OUTPUT / f"{name}.csv"
        frame.to_csv(path, index=False)
        file_meta[name] = {"rows": len(frame), "sha256": sha256(path)}

    isolation_columns = ("content_group", "split_group_id", "near_duplicate_group",
                         "audio_sha256", "source_audio_sha256", "base_audio_id",
                         "base_voice_id", "base_music_id", "parent_real_id",
                         "parent_fake_id", "voice_content_group", "music_content_group")
    isolation = {
        "train_vs_generator_val": {column: set_overlap(train, val, column)
                                    for column in isolation_columns},
        "train_vs_cal": {column: set_overlap(train, cal, column)
                          for column in isolation_columns},
    }
    generator_isolation = {
        "train_vs_generator_val": {
            component: set_overlap(
                train[train[f"{component}_fake"].eq(1)],
                val[val[f"{component}_fake"].eq(1)],
                f"{component}_generator_family")
            for component in ("voice", "music")
        }
    }
    fake_generators = set(train_core.loc[train_core.file_fake.eq(1), "generator_family"])
    voice_generators = set(train_core.loc[train_core.voice_fake.eq(1), "generator_family"])
    music_generators = set(train_core.loc[train_core.music_fake.eq(1), "generator_family"])
    paired_sources = {}
    for source, group in direct.groupby("source"):
        paired_sources[source] = {
            "rows": len(group), "content_groups": group.content_group.nunique(),
            "real": int(group.file_fake.eq(0).sum()), "fake": int(group.file_fake.eq(1).sum()),
            "voice": bool(group.voice_present.eq(1).any()),
            "music": bool(group.music_present.eq(1).any()),
        }
    paired_voice = sum(meta["voice"] and meta["real"] > 0 and meta["fake"] > 0 and
                       meta["content_groups"] >= 10 for meta in paired_sources.values())
    paired_music = sum(meta["music"] and meta["real"] > 0 and meta["fake"] > 0 and
                       meta["content_groups"] >= 10 for meta in paired_sources.values())
    rr_balance = {str(key): int(value) for key, value in mixed.mix_state.value_counts().items()}
    structural_gates = {
        "train_class_balance_reasonable": bool(0.35 <= train_core.file_fake.mean() <= 0.65),
        "voice_fake_generators_at_least_8": bool(len(voice_generators) >= 8),
        "music_fake_generators_at_least_6": bool(len(music_generators) >= 6),
        "paired_voice_sources_at_least_2": bool(paired_voice >= 2),
        "paired_music_sources_at_least_2": bool(paired_music >= 2),
        "partial_fake_positive": bool(len(partial) and partial.file_fake.eq(1).any()),
        "rr_rf_fr_ff_balanced": bool(len(set(rr_balance.values())) == 1 and set(rr_balance) == {
            "RR", "RF", "FR", "FF"}),
        "approved_metric_complete_source_disjoint_validation": False,
        "final_holdout_v13b_sealed": False,
        "split_identifier_overlap_zero": bool(
            all(value == 0 for block in isolation.values() for value in block.values()) and
            all(value == 0 for block in generator_isolation.values()
                for value in block.values())),
    }
    report = {
        "dataset_version": "DATASET_V13B_PILOT_20260901_1",
        "status": "DATASET_NOT_READY",
        "data_root": str(data_root),
        "files": file_meta,
        "direct_rows": len(direct),
        "train_rows": len(train),
        "cal_rows": len(cal),
        "generator_val_rows": len(val),
        "source_disjoint_val": "NOT CREATED: second approved paired music source missing",
        "final_holdout_v13b": "NOT CREATED / NOT SEALED: unused approved source not acquired",
        "paired_sources": paired_sources,
        "paired_voice_sources": paired_voice,
        "paired_music_sources": paired_music,
        "voice_fake_generators_train": sorted(voice_generators),
        "music_fake_generators_train": sorted(music_generators),
        "all_fake_generators_train": sorted(fake_generators),
        "partial_rows": len(partial),
        "mixed_rows": len(mixed),
        "mixed_state_counts": rr_balance,
        "channel_policy": "canonical16k_pcm16_label_independent_v1 + label-independent training augmentation",
        "isolation_audit": isolation,
        "generator_ancestry_audit": generator_isolation,
        "structural_gates": structural_gates,
        "shortcut_gate": "NOT RUN",
        "model_training": "BLOCKED until every structural and shortcut gate passes",
        "known_blockers": [key for key, passed in structural_gates.items() if not passed],
        "historical_v13_preserved": (ROOT / "data/splits_v13/DATASET_V13.json").is_file(),
    }
    (OUTPUT / "DATASET_V13B.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (data_root / "manifests/DATASET_V13B.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
