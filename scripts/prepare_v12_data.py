#!/usr/bin/env python3
"""Freeze DATASET_V12 and build a public-data-only, leakage-audited CAL_V12."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ensemble import assert_final_holdout_forbidden


SEED = 20260901
HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
SELECTED_SHA = {
    "model/best.pt": "ae354126b741f2212224da4ac6815558085ef892e32564baf2f5bb1cf326bac6",
    "model/music_best.pt": "ed87097507ed89991dd49952fbdcb9c5ceb0c256d871a385d9cdbcf9945c84c1",
    "model/fusion_weights.json": "87d46c317398bed0a9dc87c6b451246851ba6c34683e4468a0251461f7c42402",
    "script.py": "8abd4888f09aa00be18790e5257bf0cafc2fff5fd9e167571c880509a665119b",
}
OBJECTIVE = {
    "robust_total": {
        "mean_val_total": 0.25,
        "worst_val_total": 0.15,
        "voice_unseen_quality": 0.15,
        "music_unseen_quality": 0.15,
        "cal_old": 0.15,
        "cal_v12": 0.15,
    },
    "calibration_robust": {
        "cal_old_mean": 0.35,
        "cal_v12_mean": 0.35,
        "cal_old_worst": 0.15,
        "cal_v12_worst": 0.15,
    },
    "hard_gates": {
        "maximum_file_eer_regression": 0.015,
        "maximum_voice_unseen_eer_regression": 0.03,
        "maximum_music_unseen_eer_regression": 0.03,
        "maximum_worst_total_regression": 0.01,
        "minimum_bootstrap_robust_win_rate": 0.65,
    },
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else "unknown"


def select_reserved(train: pd.DataFrame, forbidden_speakers: set[str]) -> pd.DataFrame:
    eligible = train[~train["speaker_id"].astype(str).isin(forbidden_speakers)]
    mlaad = eligible[eligible["source"].eq("mlaad_tiny_matched")]
    target_voice_generators = sorted(
        generator for generator in mlaad["generator"].astype(str).unique()
        if generator != "MLAAD_original")[:4]
    selected_voice_speakers = set(
        mlaad[mlaad["generator"].isin(target_voice_generators)]["speaker_id"].astype(str))
    voice = mlaad[mlaad["speaker_id"].astype(str).isin(selected_voice_speakers)]

    echoes = eligible[eligible["source"].eq("echoes_fma_paired")]
    echoes_artists = sorted(echoes["speaker_id"].astype(str).unique())[:8]
    echoes = echoes[echoes["speaker_id"].astype(str).isin(echoes_artists)]

    jamendo = eligible[eligible["source"].eq("mtg_jamendo_cc")]
    jamendo_artists = sorted(jamendo["speaker_id"].astype(str).unique())[:50]
    jamendo = jamendo[jamendo["speaker_id"].astype(str).isin(jamendo_artists)]

    guitar = eligible[eligible["source"].eq("guitarset_mic")]
    guitar_players = sorted(guitar["speaker_id"].astype(str).unique())[:1]
    guitar = guitar[guitar["speaker_id"].astype(str).isin(guitar_players)]

    sonics = eligible[eligible["source"].eq("sonics_official")]
    sonics_generators = sorted(sonics["generator"].astype(str).unique())[:1]
    sonics = sonics[sonics["generator"].astype(str).isin(sonics_generators)]
    reserved = pd.concat([voice, echoes, jamendo, guitar, sonics]).drop_duplicates(
        subset=["path"]).reset_index(drop=True)
    if len(reserved) < 250:
        raise RuntimeError(f"insufficient public reserve rows: {len(reserved)}")
    if not reserved["allowed_for_competition"].astype(str).eq("YES").all():
        raise RuntimeError("CAL_V12 reserve contains non-approved provenance")
    return reserved


def combine_license(voice: pd.Series, music: pd.Series) -> str:
    return f"voice[{clean(voice.get('license'))}] + music[{clean(music.get('license'))}]"


def make_mixed_row(voice: pd.Series, music: pd.Series, label: str, index: int) -> dict:
    overlap = (0.25, 0.50, 0.75, 1.0)[index % 4]
    snr = (-6.0, -3.0, 0.0, 3.0, 6.0)[index % 5]
    crossfade = (0.05, 0.10, 0.25)[index % 3]
    voice_group = clean(voice.get("split_group_id"))
    music_group = clean(music.get("split_group_id"))
    original = f"calv12::{clean(voice.get('original_id'))}::{clean(music.get('original_id'))}::{index:04d}"
    digest_input = "|".join((clean(voice.get("content_hash")), clean(music.get("content_hash")),
                             str(snr), str(overlap), str(crossfade), str(index)))
    content_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    voice_fake = int(voice["voice_fake"])
    music_fake = int(music["music_fake"])
    return {
        "path": f"MIX::{voice['path']}|{music['path']}",
        "file_fake": int(voice_fake or music_fake),
        "voice_fake": voice_fake,
        "music_fake": music_fake,
        "voice_present": 1,
        "music_present": 1,
        "speaker_id": f"voice={clean(voice.get('speaker_id'))}|music={clean(music.get('speaker_id'))}",
        "generator": f"mix::{clean(voice.get('generator'))}::{clean(music.get('generator'))}",
        "source": f"cal_v12::{clean(voice.get('source'))}::{clean(music.get('source'))}",
        "dataset": "cal_v12_public_mix",
        "hf_id": f"{clean(voice.get('hf_id'))}|{clean(music.get('hf_id'))}",
        "original_id": original,
        "split_group_id": f"calv12::{voice_group}::{music_group}",
        "base_voice_id": voice_group,
        "base_music_id": music_group,
        "base_voice_speaker_id": clean(voice.get("speaker_id")),
        "base_music_speaker_id": clean(music.get("speaker_id")),
        "base_voice_original_id": clean(voice.get("original_id")),
        "base_music_original_id": clean(music.get("original_id")),
        "mix_mode": "simultaneous",
        "mix_snr_db": snr,
        "mix_overlap_fraction": overlap,
        "mix_gap_sec": 0.0,
        "mix_crossfade_sec": crossfade,
        "mix_voice_gain_db": 0.0,
        "mix_music_gain_db": 0.0,
        "calibration_fold": f"cal_v12_{'abc'[index % 3]}",
        "augment": "none",
        "source_url": f"{clean(voice.get('source_url'))}|{clean(music.get('source_url'))}",
        "version": f"{clean(voice.get('version'))}|{clean(music.get('version'))}",
        "license": combine_license(voice, music),
        "allowed_for_competition": "YES",
        "redistribution_allowed": "NO" if "NO" in {
            clean(voice.get("redistribution_allowed")),
            clean(music.get("redistribution_allowed"))} else "CONDITIONAL",
        "commercial_restriction": f"{clean(voice.get('commercial_restriction'))}|{clean(music.get('commercial_restriction'))}",
        "dataset_name": "CAL_V12 derived public mix",
        "content_hash": content_hash,
        "near_duplicate_group": f"calv12::{voice_group}::{music_group}",
        "data_role": "cal_v12",
        "upstream_split": "public_v9_train_reserve",
        "upstream_label": label,
    }


def build_calibration(reserved: pd.DataFrame) -> pd.DataFrame:
    voice = reserved[reserved["source"].eq("mlaad_tiny_matched")]
    music = reserved[reserved["source"].isin(
        ["echoes_fma_paired", "mtg_jamendo_cc", "guitarset_mic", "sonics_official"])]
    voice_real = voice[(voice["voice_present"].eq(1)) & (voice["voice_fake"].eq(0))]
    voice_fake = voice[(voice["voice_present"].eq(1)) & (voice["voice_fake"].eq(1))]
    music_real = music[(music["music_present"].eq(1)) & (music["music_fake"].eq(0))]
    music_fake = music[(music["music_present"].eq(1)) & (music["music_fake"].eq(1))]
    pools = {"RR": (voice_real, music_real), "RF": (voice_real, music_fake),
             "FR": (voice_fake, music_real), "FF": (voice_fake, music_fake)}
    if any(len(left) < 20 or len(right) < 20 for left, right in pools.values()):
        raise RuntimeError({name: (len(left), len(right)) for name, (left, right) in pools.items()})
    rng = np.random.default_rng(SEED)
    records = []
    per_class = 125
    for class_offset, (label, (left, right)) in enumerate(pools.items()):
        left_order = rng.permutation(len(left))
        right_order = rng.permutation(len(right))
        for local_index in range(per_class):
            # Coprime stepping avoids repeatedly pairing the same two source rows.
            v = left.iloc[left_order[(local_index * 7 + class_offset) % len(left)]]
            m = right.iloc[right_order[(local_index * 11 + 3 * class_offset) % len(right)]]
            records.append(make_mixed_row(v, m, label, class_offset * per_class + local_index))
    frame = pd.DataFrame(records)
    if frame["path"].duplicated().any():
        # Same source pair may be used with a different deterministic channel recipe;
        # the complete derived identity must still be unique.
        if frame[["path", "mix_snr_db", "mix_overlap_fraction", "mix_crossfade_sec"]].duplicated().any():
            raise RuntimeError("CAL_V12 contains duplicate derived mixes")
    return frame.sort_values(["calibration_fold", "upstream_label", "original_id"]).reset_index(drop=True)


def value_set(frame: pd.DataFrame, columns: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for column in columns:
        if column in frame:
            values.update(clean(value) for value in frame[column])
    values.discard("unknown")
    return values


def overlap_report(cal: pd.DataFrame, other: pd.DataFrame) -> dict:
    checks = {
        "speaker": (("base_voice_speaker_id", "base_music_speaker_id"), ("speaker_id",)),
        "original_id": (("base_voice_original_id", "base_music_original_id"), ("original_id",)),
        "split_group_id": (("base_voice_id", "base_music_id"), ("split_group_id",)),
        "base_audio_id": (("base_voice_id", "base_music_id"),
                          ("base_voice_id", "base_music_id", "split_group_id")),
        "near_duplicate": (("base_voice_id", "base_music_id"), ("near_duplicate_group", "split_group_id")),
    }
    return {name: len(value_set(cal, left) & value_set(other, right))
            for name, (left, right) in checks.items()}


def actual_component_paths(path: str) -> list[pathlib.Path]:
    if path.startswith("MIX::"):
        return [pathlib.Path(part) for part in path.split("MIX::", 1)[1].split("|", 1)]
    if path.startswith("PARTIAL::"):
        return [pathlib.Path(part) for part in path.split("PARTIAL::", 1)[1].split("|", 1)]
    return [pathlib.Path(path)]


def externalize_training_audio(train: pd.DataFrame, data_root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ROOT.resolve() == data_root.resolve() or ROOT.resolve() in data_root.resolve().parents:
        raise RuntimeError("AI_VOICE_DATA_ROOT must be outside the repository")
    files_root = data_root / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    evidence = []
    seen = set()
    for _, row in train.iterrows():
        for source_path in actual_component_paths(str(row["path"])):
            source = source_path if source_path.is_absolute() else ROOT / source_path
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            key = str(source)
            if key in mapping:
                continue
            digest = sha256(source)
            destination = files_root / f"{digest}{source.suffix.lower()}"
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            mapping[key] = str(destination.resolve())
            evidence.append({
                "path": str(destination.resolve()), "sha256": digest,
                "source": clean(row.get("source")), "license": clean(row.get("license")),
                "generator": clean(row.get("generator")),
                "split_group_id": clean(row.get("split_group_id")),
                "training_role": "v12_student_train",
            })
            seen.add(key)

    def rewrite(path: str) -> str:
        prefix = ""
        body = path
        if path.startswith("MIX::"):
            prefix, body = "MIX::", path.split("MIX::", 1)[1]
        elif path.startswith("PARTIAL::"):
            prefix, body = "PARTIAL::", path.split("PARTIAL::", 1)[1]
        parts = body.split("|", 1)
        resolved = []
        for part in parts:
            source = pathlib.Path(part)
            source = source if source.is_absolute() else ROOT / source
            resolved.append(mapping[str(source.resolve())])
        return prefix + "|".join(resolved)

    external = train.copy()
    external["path"] = external["path"].astype(str).map(rewrite)
    return external, pd.DataFrame(evidence).sort_values("path").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get(
        "AI_VOICE_DATA_ROOT", str(pathlib.Path.home() / "ai_voice_data_v12")))
    args = parser.parse_args()
    output_dir = ROOT / "data/splits_v12"
    experiment_dir = ROOT / "experiments/v12"
    for path in (output_dir, experiment_dir, args.data_root):
        assert_final_holdout_forbidden(path)
    for relative, expected in SELECTED_SHA.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"selected V7 freeze mismatch {relative}: {actual} != {expected}")

    source_train = pd.read_csv(ROOT / "data/splits_v9_candidate/train.csv")
    validation_frames = [pd.read_csv(ROOT / f"data/splits_v9_candidate/{name}.csv")
                         for name in ("val_a", "val_b", "val_c", "val_d")]
    forbidden_speakers = set(pd.concat(validation_frames, ignore_index=True)[
        "speaker_id"].astype(str))
    reserved = select_reserved(source_train, forbidden_speakers)
    reserve_paths = set(reserved["path"].astype(str))
    train = source_train[~source_train["path"].astype(str).isin(reserve_paths)].copy()
    train["data_role"] = "train_v12"
    cal = build_calibration(reserved)

    validation = pd.concat(validation_frames, ignore_index=True)
    expanded = validation_frames[1][validation_frames[1]["data_role"].isin(
        ["val_b_unseen_generator", "val_b_unseen_music_generator"])]
    overlaps = {
        "train": overlap_report(cal, train),
        "validation": overlap_report(cal, validation),
        "expanded_unseen": overlap_report(cal, expanded),
        "final_holdout": "NOT READ / NOT RUN; inherited V9 public-data isolation audit",
    }
    numeric_overlaps = [value for role in ("train", "validation", "expanded_unseen")
                        for value in overlaps[role].values()]
    if any(numeric_overlaps):
        raise RuntimeError(f"CAL_V12 leakage detected: {overlaps}")

    data_root = pathlib.Path(args.data_root).resolve()
    external_train, used_files = externalize_training_audio(train, data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    external_train.to_csv(output_dir / "train.csv", index=False)
    cal.to_csv(output_dir / "cal_v12.csv", index=False)
    reserved.to_csv(output_dir / "cal_v12_base_reserve.csv", index=False)
    used_files.to_csv(output_dir / "used_training_files_v12.csv", index=False)
    manifest = pd.concat([external_train, cal], ignore_index=True, sort=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    for name in ("val_a", "val_b", "val_c", "val_d"):
        shutil.copy2(ROOT / f"data/splits_v9_candidate/{name}.csv", output_dir / f"{name}.csv")
    shutil.copy2(ROOT / "data/splits/fusion_calibration.csv", output_dir / "cal_old.csv")

    source_counts = cal["source"].str.split("::", expand=True)
    report = {
        "status": "PASS",
        "final_holdout": "NOT READ / NOT RUN",
        "rows": len(cal),
        "class_balance": cal["upstream_label"].value_counts().sort_index().to_dict(),
        "fold_balance": cal["calibration_fold"].value_counts().sort_index().to_dict(),
        "voice_generators": sorted(set(reserved.loc[
            reserved["source"].eq("mlaad_tiny_matched"), "generator"].astype(str))),
        "music_generators": sorted(set(reserved.loc[
            reserved["source"].isin(["echoes_fma_paired", "sonics_official"]),
            "generator"].astype(str))),
        "real_sources": sorted(set(reserved.loc[
            (reserved["voice_fake"].eq(0)) | (reserved["music_fake"].eq(0)), "source"])),
        "reserved_base_rows": len(reserved),
        "v12_training_rows": len(external_train),
        "external_data_root": str(data_root),
        "used_training_audio_files": len(used_files),
        "overlap": overlaps,
    }
    (experiment_dir / "cal_v12_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (experiment_dir / "selection_policy_precommitted.json").write_text(
        json.dumps(OBJECTIVE, indent=2) + "\n", encoding="utf-8")

    split_hashes = {path.name: sha256(path) for path in sorted(output_dir.glob("*.csv"))}
    dataset = {
        "dataset_version": "DATASET_V12_20260901_2",
        "manifest_sha256": sha256(output_dir / "manifest.csv"),
        "split_sha256": split_hashes,
        "training_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "selected_v7_sha256": SELECTED_SHA,
        "data_root": str(data_root),
        "final_holdout": "NOT READ / NOT RUN",
        "selection_policy_sha256": sha256(experiment_dir / "selection_policy_precommitted.json"),
    }
    (output_dir / "DATASET_V12.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
