#!/usr/bin/env python3
"""Validate and stage a 10–100-row-group paired-music pilot manifest.

This tool never downloads media and never infers approval from a dataset-level
license. Every real/fake row must carry explicit row-level rights evidence and
both members of each content pair must already exist locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import pandas as pd


REQUIRED = (
    "path", "file_fake", "voice_fake", "music_fake", "voice_present",
    "music_present", "source", "dataset", "source_url", "license",
    "competition_use_status", "approval_basis", "license_source",
    "license_snapshot_sha256", "reviewed_at", "original_id", "content_group",
    "generator_family", "split_group_id",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=10)
    parser.add_argument("--maximum-pairs", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise RuntimeError(f"paired music manifest missing columns: {missing}")
    if not frame.competition_use_status.eq("APPROVED").all():
        raise RuntimeError("REVIEW_REQUIRED/unknown rows cannot enter the pilot")
    for column in ("source_url", "license", "approval_basis", "license_source",
                   "license_snapshot_sha256", "reviewed_at"):
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            raise RuntimeError(f"row-level provenance field is empty: {column}")
    if not (frame.voice_present.eq(0) & frame.music_present.eq(1)).all():
        raise RuntimeError("paired music pilot rows must be music-present and voice-absent")
    if not frame.file_fake.eq(frame.music_fake).all():
        raise RuntimeError("file_fake must equal music_fake for the paired music pilot")
    pairs = frame.groupby("content_group").file_fake.agg(set)
    if not pairs.map(lambda labels: labels == {0, 1}).all():
        raise RuntimeError("every content_group requires both real and fake rows")
    if not args.minimum_pairs <= len(pairs) <= args.maximum_pairs:
        raise RuntimeError(
            f"pilot must contain {args.minimum_pairs}–{args.maximum_pairs} pairs; got {len(pairs)}")
    resolved = []
    hashes = []
    for value in frame.path:
        path = pathlib.Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved.append(str(path))
        hashes.append(sha256(path))
    frame["path"] = resolved
    frame["audio_sha256"] = hashes
    if frame.audio_sha256.duplicated().any():
        raise RuntimeError("exact duplicate audio detected in paired music pilot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    report = {
        "status": "PILOT_STAGED_NOT_PRODUCTION_APPROVED",
        "rows": len(frame), "pairs": len(pairs),
        "sources": sorted(frame.source.unique()),
        "generators": sorted(frame.loc[frame.file_fake.eq(1), "generator_family"].unique()),
        "manifest_sha256": sha256(args.output),
        "next_gate": "grouped shortcut audit + source-disjoint unseen EER before scaling",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
