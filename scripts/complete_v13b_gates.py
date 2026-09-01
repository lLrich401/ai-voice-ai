#!/usr/bin/env python3
"""Validate optional V13B adoption data without weakening exploratory gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANCESTRY = ("content_group", "split_group_id", "near_duplicate_group", "base_audio_id",
            "base_voice_id", "base_music_id", "parent_real_id", "parent_fake_id",
            "voice_content_group", "music_content_group", "audio_sha256",
            "source_audio_sha256")
PROVENANCE = ("source_url", "license", "approval_basis", "license_source",
              "license_snapshot_sha256", "reviewed_at")
IGNORE = {"", "nan", "none", "ABSENT", "VIRTUAL", "REAL_CONTROL"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    result: set[str] = set()
    for raw in frame[column].dropna().astype(str):
        result.update(value for value in raw.split("|") if value not in IGNORE)
    return result


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], role: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{role} manifest missing columns: {missing}")


def require_approved(frame: pd.DataFrame, role: str) -> None:
    require_columns(frame, ("competition_use_status",) + PROVENANCE, role)
    if not frame.competition_use_status.eq("APPROVED").all():
        raise RuntimeError(f"{role} contains non-APPROVED rows")
    for column in PROVENANCE:
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{role} has empty provenance field: {column}")


def stable_content_role(content_group: str) -> str:
    bucket = int(hashlib.sha256(str(content_group).encode()).hexdigest()[:8], 16) % 10
    return "cal_v13b" if bucket < 2 else "train"


def validate_paired_music(candidate: pd.DataFrame, existing: pd.DataFrame,
                          minimum_groups: int = 10) -> dict:
    required = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present",
                "source", "content_group", "generator_family", "split_group_id")
    require_columns(candidate, required, "paired music")
    require_approved(candidate, "paired music")
    if not (candidate.voice_present.eq(0) & candidate.music_present.eq(1)).all():
        raise RuntimeError("paired music rows must be voice-absent/music-present")
    if not candidate.file_fake.eq(candidate.music_fake).all():
        raise RuntimeError("paired music file_fake must equal music_fake")
    labels = candidate.groupby("content_group").file_fake.agg(set)
    if len(labels) < minimum_groups or not labels.map(lambda value: value == {0, 1}).all():
        raise RuntimeError(f"paired music needs at least {minimum_groups} complete real/fake groups")
    existing_music_sources = set(existing.loc[existing.music_present.eq(1), "source"].astype(str))
    overlap = sorted(set(candidate.source.astype(str)) & existing_music_sources)
    if overlap:
        raise RuntimeError(f"second paired music source is not independent: {overlap}")
    result = candidate.copy()
    result["v13b_role"] = result.content_group.map(stable_content_role)
    return {
        "status": "PASS_PILOT", "rows": len(result), "content_groups": len(labels),
        "sources": sorted(result.source.astype(str).unique()),
        "role_counts": {str(key): int(value) for key, value in result.v13b_role.value_counts().items()},
        "frame": result,
    }


def metric_complete(frame: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "file_real_fake": set(frame.file_fake.astype(int)) == {0, 1},
        "voice_presence_0_1": set(frame.voice_present.astype(int)) == {0, 1},
        "music_presence_0_1": set(frame.music_present.astype(int)) == {0, 1},
        "voice_real_fake_when_present": set(
            frame.loc[frame.voice_present.eq(1), "voice_fake"].astype(int)) == {0, 1},
        "music_real_fake_when_present": set(
            frame.loc[frame.music_present.eq(1), "music_fake"].astype(int)) == {0, 1},
    }
    return checks


def overlap_report(candidate: pd.DataFrame, reference: pd.DataFrame,
                   include_source: bool, include_generator: bool) -> dict[str, int]:
    columns = list(ANCESTRY)
    if include_source:
        columns.extend(("source", "dataset"))
    result = {column: len(tokens(candidate, column) & tokens(reference, column))
              for column in columns}
    if include_generator:
        for component in ("voice", "music"):
            column = f"{component}_generator_family"
            fake = f"{component}_fake"
            left = candidate[candidate[fake].eq(1)] if fake in candidate else candidate.iloc[0:0]
            right = reference[reference[fake].eq(1)] if fake in reference else reference.iloc[0:0]
            result[column] = len(tokens(left, column) & tokens(right, column))
    return result


def validate_source_disjoint(candidate: pd.DataFrame, development: pd.DataFrame) -> dict:
    require_columns(candidate, ("file_fake", "voice_fake", "music_fake", "voice_present",
                                "music_present", "source", "content_group"), "source-disjoint")
    require_approved(candidate, "source-disjoint")
    completeness = metric_complete(candidate)
    if not all(completeness.values()):
        raise RuntimeError(f"source-disjoint manifest is not metric-complete: {completeness}")
    overlap = overlap_report(candidate, development, include_source=True, include_generator=False)
    if any(overlap.values()):
        raise RuntimeError(f"source-disjoint overlap detected: {overlap}")
    return {"status": "PASS", "rows": len(candidate), "metric_complete": completeness,
            "overlap": overlap}


def history_frames() -> tuple[pd.DataFrame, list[str]]:
    frames, paths = [], []
    for base in (ROOT / "data", ROOT / "experiments"):
        for path in sorted(base.rglob("*.csv")):
            if "final_holdout_v13b" in path.name.lower():
                continue
            try:
                frame = pd.read_csv(path, low_memory=False)
            except (pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
                continue
            if {"source", "file_fake"} & set(frame.columns):
                frames.append(frame)
                paths.append(path.relative_to(ROOT).as_posix())
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(), paths


def validate_final(candidate: pd.DataFrame) -> dict:
    require_approved(candidate, "final holdout")
    completeness = metric_complete(candidate)
    if not all(completeness.values()):
        raise RuntimeError(f"final holdout is not metric-complete: {completeness}")
    history, paths = history_frames()
    overlap = overlap_report(candidate, history, include_source=True, include_generator=True)
    if any(overlap.values()):
        raise RuntimeError(f"final holdout overlaps global project history: {overlap}")
    return {"status": "SEALED_NOT_FOR_DEVELOPMENT", "rows": len(candidate),
            "history_files_scanned": len(paths), "overlap": overlap,
            "maximum_evaluations": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-music", type=pathlib.Path)
    parser.add_argument("--source-disjoint", type=pathlib.Path)
    parser.add_argument("--final-holdout", type=pathlib.Path)
    parser.add_argument("--data-root", default=os.getenv("AI_VOICE_DATA_ROOT"))
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "data/splits_v13b/train.csv")
    cal = pd.read_csv(ROOT / "data/splits_v13b/cal_v13b.csv")
    development = pd.concat([train, cal], ignore_index=True, sort=False)
    report = {"paired_music": "NOT ACQUIRED", "source_disjoint": "NOT ACQUIRED",
              "final_holdout": "NOT ACQUIRED / NOT SEALED / NOT RUN"}
    if args.paired_music:
        validated = validate_paired_music(pd.read_csv(args.paired_music), development)
        frame = validated.pop("frame")
        if not args.data_root:
            raise RuntimeError("AI_VOICE_DATA_ROOT/--data-root required for staged data")
        destination = pathlib.Path(args.data_root).resolve() / "splits/paired_music_v13b_pilot.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        report["paired_music"] = {**validated, "path": str(destination), "sha256": sha256(destination)}
    if args.source_disjoint:
        report["source_disjoint"] = validate_source_disjoint(
            pd.read_csv(args.source_disjoint), development)
    if args.final_holdout:
        if not args.data_root:
            raise RuntimeError("AI_VOICE_DATA_ROOT/--data-root required for final holdout")
        source = args.final_holdout.resolve()
        final = pd.read_csv(source)
        seal = validate_final(final)
        destination = pathlib.Path(args.data_root).resolve() / "splits/final_holdout_v13b.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(destination, index=False)
        seal.update({"sha256": sha256(destination), "external_path": str(destination)})
        seal_path = ROOT / "experiments/v13b/FINAL_HOLDOUT_V13B_SEAL.json"
        seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
        report["final_holdout"] = seal
    output = ROOT / "experiments/v13b/gate_completion_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
