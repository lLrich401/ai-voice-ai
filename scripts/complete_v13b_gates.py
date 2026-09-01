#!/usr/bin/env python3
"""Validate optional V13B adoption data without weakening exploratory gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_global_history_index_v13b import (
    build_global_history_index,
    overlap_with_index,
)


LABELS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
ANCESTRY = ("content_group", "split_group_id", "near_duplicate_group", "base_audio_id",
            "base_voice_id", "base_music_id", "parent_real_id", "parent_fake_id",
            "voice_content_group", "music_content_group", "audio_sha256",
            "source_audio_sha256")
GENERATOR_LINEAGE = ("generator_family", "voice_generator_family", "music_generator_family")
IDENTITY = ("source", "dataset") + ANCESTRY + GENERATOR_LINEAGE
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


def require_identity_values(frame: pd.DataFrame, role: str) -> None:
    """Unknown identity is a validation failure, never evidence of zero overlap."""
    require_columns(frame, IDENTITY, role)
    for column in IDENTITY:
        values = frame[column].fillna("").astype(str).str.strip()
        if values.eq("").any() or values.str.lower().eq("nan").any():
            raise RuntimeError(f"{role} has unknown required identity: {column}")
    for component in ("voice", "music"):
        fake = f"{component}_fake"
        lineage = f"{component}_generator_family"
        invalid = frame.loc[frame[fake].astype(int).eq(1), lineage].astype(str).str.strip()
        if invalid.isin(IGNORE).any() or invalid.str.lower().eq("nan").any():
            raise RuntimeError(f"{role} has unknown fake {component} generator lineage")


def require_isolation_manifest(frame: pd.DataFrame, role: str) -> None:
    require_columns(frame, LABELS, role)
    require_identity_values(frame, role)
    require_approved(frame, role)


def stable_content_role(content_group: str) -> str:
    bucket = int(hashlib.sha256(str(content_group).encode()).hexdigest()[:8], 16) % 10
    return "cal_v13b" if bucket < 2 else "train"


def validate_paired_music(candidate: pd.DataFrame, existing: pd.DataFrame,
                          minimum_groups: int = 10) -> dict:
    require_isolation_manifest(candidate, "paired music")
    if not (candidate.voice_present.eq(0) & candidate.music_present.eq(1)).all():
        raise RuntimeError("paired music rows must be voice-absent/music-present")
    if not candidate.file_fake.eq(candidate.music_fake).all():
        raise RuntimeError("paired music file_fake must equal music_fake")
    labels = candidate.groupby("content_group").file_fake.agg(set)
    if len(labels) < minimum_groups or not labels.map(lambda value: value == {0, 1}).all():
        raise RuntimeError(f"paired music needs at least {minimum_groups} complete real/fake groups")
    overlap = overlap_report(candidate, existing, include_source=True, include_generator=False)
    if any(overlap.values()):
        raise RuntimeError(f"paired music overlaps existing development history: {overlap}")
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
    require_isolation_manifest(candidate, "source-disjoint")
    completeness = metric_complete(candidate)
    if not all(completeness.values()):
        raise RuntimeError(f"source-disjoint manifest is not metric-complete: {completeness}")
    overlap = overlap_report(candidate, development, include_source=True, include_generator=False)
    if any(overlap.values()):
        raise RuntimeError(f"source-disjoint overlap detected: {overlap}")
    generator_overlap = overlap_report(candidate, development, include_source=False,
                                       include_generator=True)
    return {"status": "PASS", "rows": len(candidate), "metric_complete": completeness,
            "overlap": overlap, "generator_lineage_overlap_observed": {
                key: value for key, value in generator_overlap.items()
                if key in GENERATOR_LINEAGE},
            "unknown_identity_is_fail": True}


def validate_final(candidate: pd.DataFrame, history_index: dict | None = None,
                   *, external_history_configured: bool = False) -> dict:
    require_isolation_manifest(candidate, "final holdout")
    completeness = metric_complete(candidate)
    if not all(completeness.values()):
        raise RuntimeError(f"final holdout is not metric-complete: {completeness}")
    if not external_history_configured:
        raise RuntimeError("FINAL_VALIDATION_FAIL: external history root is not configured")
    history_index = history_index or build_global_history_index()
    if (history_index.get("status") != "PASS" or not history_index.get("files_scanned") or
            not history_index.get("entries") or not history_index.get("external_root_scanned")):
        raise RuntimeError("FINAL_VALIDATION_FAIL: global history index is incomplete")
    overlap = overlap_with_index(candidate, history_index, include_generator=True)
    if any(overlap.values()):
        raise RuntimeError(f"final holdout overlaps global project history: {overlap}")
    return {"status": "SEALED_NOT_FOR_DEVELOPMENT", "rows": len(candidate),
            "history_files_scanned": history_index["history_files_scanned"],
            "history_entries": history_index["history_entries"],
            "external_root_scanned": history_index["external_root_scanned"], "overlap": overlap,
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
    generator_val = pd.read_csv(ROOT / "data/splits_v13b/val_generator_disjoint.csv")
    development = pd.concat([train, cal, generator_val], ignore_index=True, sort=False)
    root = pathlib.Path(args.data_root).resolve() if args.data_root else None
    report = {"paired_music": "NOT ACQUIRED", "source_disjoint": "NOT ACQUIRED",
              "final_holdout": "NOT ACQUIRED / NOT SEALED / NOT RUN"}
    added_history: dict[str, pd.DataFrame] = {}
    if args.paired_music:
        validated = validate_paired_music(pd.read_csv(args.paired_music), development)
        frame = validated.pop("frame")
        if not args.data_root:
            raise RuntimeError("AI_VOICE_DATA_ROOT/--data-root required for staged data")
        destination = pathlib.Path(args.data_root).resolve() / "splits/paired_music_v13b_pilot.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        development = pd.concat([development, frame], ignore_index=True, sort=False)
        added_history["paired_music_v13b_pilot"] = frame
        report["paired_music"] = {**validated, "path": str(destination), "sha256": sha256(destination)}
    if args.source_disjoint:
        source_disjoint = pd.read_csv(args.source_disjoint)
        report["source_disjoint"] = validate_source_disjoint(source_disjoint, development)
        # It is never appended to train, but future FINAL checks must see that
        # it was used for candidate/model selection.
        added_history["source_disjoint_model_selection"] = source_disjoint
    history_index = build_global_history_index(root, extra_frames=added_history)
    if history_index["status"] != "PASS":
        raise RuntimeError("global history index could not be built")
    history_path = ROOT / "experiments/v13b/global_history_index.json"
    history_path.write_text(json.dumps(history_index, indent=2) + "\n", encoding="utf-8")
    if args.final_holdout:
        if not args.data_root:
            raise RuntimeError("AI_VOICE_DATA_ROOT/--data-root required for final holdout")
        source = args.final_holdout.resolve()
        final = pd.read_csv(source)
        seal = validate_final(final, history_index, external_history_configured=root is not None)
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
