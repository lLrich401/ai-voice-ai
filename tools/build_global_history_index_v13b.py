#!/usr/bin/env python3
"""Build a conservative index of audio identities used anywhere in V13B history."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections import defaultdict

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
IDENTITY_COLUMNS = (
    "source", "dataset", "generator_family", "generator",
    "content_group", "split_group_id", "near_duplicate_group",
    "base_audio_id", "base_voice_id", "base_music_id",
    "parent_real_id", "parent_fake_id", "voice_content_group",
    "music_content_group", "audio_sha256", "source_audio_sha256",
)
IGNORE = {"", "nan", "none", "ABSENT", "VIRTUAL", "REAL_CONTROL"}


def _tokens(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    result: set[str] = set()
    for raw in frame[column].dropna().astype(str):
        result.update(value for value in raw.split("|") if value not in IGNORE)
    return result


def scan_csv_paths(data_root: pathlib.Path | None = None) -> list[pathlib.Path]:
    paths: set[pathlib.Path] = set()
    for root in (ROOT / "data", ROOT / "experiments"):
        if root.is_dir():
            paths.update(root.rglob("*.csv"))
    paths.update(ROOT.glob("training_evidence_*/**/*.csv"))
    if data_root is not None:
        for relative in ("manifests", "provenance", "splits", "training_evidence"):
            root = data_root / relative
            if root.is_dir():
                paths.update(root.rglob("*.csv"))
    return sorted(path.resolve() for path in paths
                  if "final_holdout_v13b" not in path.name.lower())


def role_for(path: pathlib.Path, frame: pd.DataFrame) -> set[str]:
    roles = {path.parent.name}
    for column in ("v13b_role", "data_role", "split", "role"):
        roles.update(_tokens(frame, column))
    return {value for value in roles if value}


def build_global_history_index(data_root: pathlib.Path | None = None) -> dict:
    entries: dict[tuple[str, str], dict] = {}
    files_scanned: list[str] = []
    skipped: list[str] = []
    for path in scan_csv_paths(data_root):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (pd.errors.EmptyDataError, UnicodeDecodeError, ValueError, OSError):
            skipped.append(str(path))
            continue
        if not set(frame.columns) & set(IDENTITY_COLUMNS):
            continue
        files_scanned.append(str(path))
        roles = role_for(path, frame)
        # Index per source/dataset *row group*.  Aggregating every identity in a
        # CSV into every source would hide provenance relationships and create
        # misleading provenance evidence (even though it is conservative for
        # overlap detection).
        source_values = (frame["source"].fillna("<SOURCE_MISSING>").astype(str)
                         if "source" in frame else pd.Series("<SOURCE_MISSING>", index=frame.index))
        dataset_values = (frame["dataset"].fillna("<DATASET_MISSING>").astype(str)
                          if "dataset" in frame else pd.Series("<DATASET_MISSING>", index=frame.index))
        grouped = frame.assign(_history_source=source_values, _history_dataset=dataset_values)
        for (source, dataset), subset in grouped.groupby(["_history_source", "_history_dataset"], dropna=False):
            key = (str(source), str(dataset))
            entry = entries.setdefault(key, {
                "source": str(source), "dataset": str(dataset), "rows_observed": 0,
                "roles": set(), "first_seen_file": str(path),
                "first_seen_version": path.parent.name,
                "identities": {column: set() for column in IDENTITY_COLUMNS},
            })
            entry["rows_observed"] += len(subset)
            entry["roles"].update(roles)
            for column in IDENTITY_COLUMNS:
                entry["identities"][column].update(_tokens(subset, column))
    materialized = []
    for entry in entries.values():
        materialized.append({
            "source": entry["source"], "dataset": entry["dataset"],
            "rows_observed": entry["rows_observed"], "roles": sorted(entry["roles"]),
            "first_seen_version": entry["first_seen_version"],
            "first_seen_file": entry["first_seen_file"],
            "identities": {key: sorted(value) for key, value in entry["identities"].items()},
        })
    return {
        "version": "V13B_GLOBAL_HISTORY_INDEX_V1",
        "status": "PASS",
        "files_scanned": files_scanned,
        "external_data_root": str(data_root) if data_root else "NOT_CONFIGURED",
        "skipped_files": skipped,
        "entries": sorted(materialized, key=lambda value: (value["source"], value["dataset"])),
    }


def index_tokens(index: dict, column: str) -> set[str]:
    values: set[str] = set()
    for entry in index["entries"]:
        values.update(entry["identities"].get(column, []))
    return values


def overlap_with_index(candidate: pd.DataFrame, index: dict, *, include_generator: bool) -> dict[str, int]:
    columns = list(IDENTITY_COLUMNS)
    result = {column: len(_tokens(candidate, column) & index_tokens(index, column))
              for column in columns}
    if include_generator:
        for component in ("voice", "music"):
            column = f"{component}_generator_family"
            fake = f"{component}_fake"
            rows = candidate[candidate[fake].eq(1)] if fake in candidate else candidate.iloc[0:0]
            result[column] = len(_tokens(rows, column) & index_tokens(index, column))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.getenv("AI_VOICE_DATA_ROOT"))
    parser.add_argument("--output", default="experiments/v13b/global_history_index.json")
    args = parser.parse_args()
    root = pathlib.Path(args.data_root).resolve() if args.data_root else None
    payload = build_global_history_index(root)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files": len(payload["files_scanned"]),
                      "entries": len(payload["entries"])}, indent=2))


if __name__ == "__main__":
    main()
