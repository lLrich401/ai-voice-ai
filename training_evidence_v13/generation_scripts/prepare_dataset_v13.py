#!/usr/bin/env python3
"""Build a provenance-first V13 pilot and seal a source-disjoint holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = pathlib.Path.home() / "ai_voice_data_v13"
OUTPUT = ROOT / "data/splits_v13"

FINAL_SOURCES = {"wavefake_ajay", "echoes_fma_paired"}
APPROVED_SOURCE_LICENSES = {
    "librispeech_dev": ("CC BY 4.0", "APPROVED"),
    "project_procedural_v8": ("project-authored numeric synthesis", "APPROVED"),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def column_or(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    """Return an index-aligned Series even when an optional column is absent."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def resolve(path_value: object) -> pathlib.Path:
    path = pathlib.Path(str(path_value))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def status(row: pd.Series) -> tuple[str, str]:
    source = str(row.source)
    if source in APPROVED_SOURCE_LICENSES:
        return APPROVED_SOURCE_LICENSES[source]
    if str(row.get("allowed_for_competition", "")).strip().upper() == "YES":
        return str(row.get("license", "recorded upstream terms")), "APPROVED"
    return str(row.get("license", "UNKNOWN")), "REVIEW_REQUIRED"


def fake_generator(row: pd.Series) -> str:
    return str(row.generator) if int(row.file_fake) else "REAL"


def assign_mlaad(frame: pd.DataFrame) -> pd.Series:
    fake = frame[frame.voice_fake.eq(1)]
    generators = sorted(fake.generator.unique())
    if len(generators) < 12:
        raise RuntimeError("MLAAD pilot requires at least 12 fake generator families")
    train, val = set(generators[:8]), set(generators[8:11])
    cal = set(generators[11:])
    group_generator = fake.groupby("split_group_id").generator.first().to_dict()
    roles = []
    for _, row in frame.iterrows():
        generator = group_generator.get(row.split_group_id)
        roles.append("train" if generator in train else
                     "val_generator_disjoint" if generator in val else
                     "cal_v13" if generator in cal else "exclude")
    return pd.Series(roles, index=frame.index)


def assign_sonics(frame: pd.DataFrame) -> pd.Series:
    generators = sorted(frame.generator.unique())
    train, val = set(generators[:3]), set(generators[3:4])
    return frame.generator.map(
        lambda value: "train" if value in train else
        "val_generator_disjoint" if value in val else "cal_v13")


def assign_guitar(frame: pd.DataFrame) -> pd.Series:
    groups = sorted(frame.split_group_id.unique())
    train, val = set(groups[:4]), set(groups[4:5])
    return frame.split_group_id.map(
        lambda value: "train" if value in train else
        "val_generator_disjoint" if value in val else "cal_v13")


def assign_roles(frame: pd.DataFrame) -> pd.Series:
    roles = pd.Series("exclude", index=frame.index, dtype=object)
    roles[frame.source.isin(FINAL_SOURCES)] = "final_holdout_v13"
    for source, assigner in (
        ("mlaad_tiny_matched", assign_mlaad),
        ("sonics_official", assign_sonics),
        ("guitarset_mic", assign_guitar),
    ):
        mask = frame.source.eq(source)
        roles.loc[mask] = assigner(frame.loc[mask])
    roles[frame.source.eq("librispeech_dev")] = "train"
    roles[frame.source.eq("project_procedural_v8") &
          frame.get("recommended_split", pd.Series("", index=frame.index)).eq("train_candidate")] = "train"
    roles[frame.source.isin({"asvspoof2019", "fake_music_generated", "gtzan_real_v2"})] = (
        "val_source_disjoint_review")
    roles[frame.source.eq("mtg_jamendo_cc")] = "cal_v13"
    return roles


def hardlink(row: pd.Series, data_root: pathlib.Path) -> tuple[str, str]:
    source = resolve(row.path)
    digest = str(row.get("content_hash", ""))
    if not digest or digest.lower() == "nan":
        digest = str(row.get("audio_sha256", ""))
    if not digest or digest.lower() == "nan":
        digest = sha256(source)
    suffix = source.suffix.lower() or ".audio"
    destination = data_root / "raw" / str(row.source) / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return str(destination), digest


def overlaps(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    result = {}
    for column in ("source", "split_group_id", "original_id", "content_group",
                   "near_duplicate_group", "sha256"):
        a = set(left[column].dropna().astype(str)) - {"", "nan"}
        b = set(right[column].dropna().astype(str)) - {"", "nan"}
        result[column] = len(a & b)
    fake_left = set(left.loc[left.file_fake.eq(1), "generator_family"].astype(str))
    fake_right = set(right.loc[right.file_fake.eq(1), "generator_family"].astype(str))
    result["fake_generator_family"] = len(fake_left & fake_right)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.getenv("AI_VOICE_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    args = parser.parse_args()
    data_root = pathlib.Path(args.data_root).resolve()
    for directory in ("raw", "generated", "processed", "paired", "partial", "mixed",
                      "channel_augmented", "manifests", "splits", "provenance", "licenses",
                      "training_runs", "frozen_versions"):
        (data_root / directory).mkdir(parents=True, exist_ok=True)

    public = pd.read_csv(ROOT / "data/splits_v9_candidate/manifest.csv")
    procedural = pd.read_csv(ROOT / "data/generated_v8/manifest.csv")
    source = pd.concat([public, procedural], ignore_index=True, sort=False)
    source["v13_role"] = assign_roles(source)
    source = source[~source.v13_role.eq("exclude")].copy().reset_index(drop=True)
    external_paths, hashes, licenses, statuses = [], [], [], []
    for _, row in source.iterrows():
        external, digest = hardlink(row, data_root)
        license_name, use_status = status(row)
        external_paths.append(external)
        hashes.append(digest)
        licenses.append(license_name)
        statuses.append(use_status)
    source["source_path"] = source.path.astype(str)
    source["path"] = external_paths
    source["sha256"] = hashes
    source["license"] = licenses
    source["competition_use_status"] = statuses
    source["generator_family"] = source.apply(fake_generator, axis=1)
    source["generator_version"] = column_or(source, "version", "unknown").fillna("unknown")
    source["language"] = source.source_path.str.extract(r"[\\/](ko|en|de|fr|es|it|pl|ru|uk)[\\/]", expand=False).fillna("unknown")
    source["speaker"] = column_or(source, "speaker_id", "unknown").fillna("unknown")
    source["track"] = column_or(source, "original_id", "unknown").fillna("unknown")
    source["content_group"] = column_or(source, "original_id", source.split_group_id).fillna(source.split_group_id)
    source["split_group"] = source.split_group_id
    source["augmentation"] = column_or(source, "augment", "none").fillna("none")
    source["seed"] = column_or(source, "generation_seed", -1).fillna(-1).astype(int)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    dev = source[~source.v13_role.eq("final_holdout_v13")].copy()
    sealed = source[source.v13_role.eq("final_holdout_v13")].copy()
    roles = {}
    for role, frame in dev.groupby("v13_role", sort=True):
        path = OUTPUT / f"{role}.csv"
        frame.to_csv(path, index=False)
        roles[role] = {"rows": len(frame), "sha256": sha256(path),
                       "approved": int(frame.competition_use_status.eq("APPROVED").sum())}
        approved = frame[frame.competition_use_status.eq("APPROVED")]
        approved.to_csv(OUTPUT / f"{role}_approved.csv", index=False)
    dev.to_csv(OUTPUT / "manifest.csv", index=False)
    sealed_path = OUTPUT / "final_holdout_v13.csv"
    sealed.to_csv(sealed_path, index=False)
    seal = {"path": "data/splits_v13/final_holdout_v13.csv", "rows": len(sealed),
            "sha256": sha256(sealed_path), "status": "SEALED_NOT_FOR_DEVELOPMENT"}
    (OUTPUT / "FINAL_HOLDOUT_V13_SEAL.json").write_text(
        json.dumps(seal, indent=2) + "\n", encoding="utf-8")

    train = dev[dev.v13_role.eq("train")]
    audit = {
        "train_vs_val_generator": overlaps(
            train, dev[dev.v13_role.eq("val_generator_disjoint")]),
        "train_vs_val_source": overlaps(
            train, dev[dev.v13_role.eq("val_source_disjoint_review")]),
        "train_vs_cal": overlaps(train, dev[dev.v13_role.eq("cal_v13")]),
        "train_vs_final": overlaps(train, sealed),
        "all_dev_vs_final": overlaps(dev, sealed),
    }
    if any(audit["all_dev_vs_final"].values()):
        raise RuntimeError(f"V13 final holdout leakage: {audit['all_dev_vs_final']}")
    report = {
        "dataset_version": "DATASET_V13_PILOT_20260901_2",
        "status": "PILOT_READY_WITH_GAPS",
        "data_root": str(data_root),
        "manifest_sha256": sha256(OUTPUT / "manifest.csv"),
        "roles": roles,
        "sealed_final": seal,
        "overlap_audit": audit,
        "known_gaps": [
            "approved source-disjoint development validation is not metric-complete",
            "music training lacks a second content-matched real/fake source after Echoes is sealed",
            "voice and music row counts remain far below V13 scaling targets",
            "REVIEW_REQUIRED rows are excluded from approved split files",
        ],
        "final_holdout": "CREATED AND SEALED; METRICS NOT RUN",
    }
    (OUTPUT / "DATASET_V13.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
