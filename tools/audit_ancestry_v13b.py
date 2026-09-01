#!/usr/bin/env python3
"""Fail-closed V13B split audit using direct and virtual-sample ancestry."""

from __future__ import annotations

import json
import pathlib

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits_v13b"
OUT = ROOT / "experiments/v13b/ancestry_leakage_audit.json"
HISTORY_OUT = ROOT / "experiments/v13b/global_history_holdout_audit.json"

ANCESTRY = (
    "base_audio_id", "base_voice_id", "base_music_id", "parent_real_id",
    "parent_fake_id", "voice_content_group", "music_content_group",
)
CONTENT = ("content_group", "split_group_id", "near_duplicate_group")
HASHES = ("audio_sha256", "source_audio_sha256")
GENERATORS = ("voice_generator_family", "music_generator_family")
IGNORED = {"", "nan", "none", "VIRTUAL", "ABSENT", "REAL_CONTROL"}


def read_optional(name: str) -> pd.DataFrame | None:
    path = SPLITS / name
    return pd.read_csv(path) if path.is_file() else None


def tokens(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    output: set[str] = set()
    for raw in frame[column].dropna().astype(str):
        output.update(part for part in raw.split("|") if part not in IGNORED)
    return output


def overlaps(left: pd.DataFrame, right: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict]:
    result = {}
    for column in columns:
        common = sorted(tokens(left, column) & tokens(right, column))
        result[column] = {"count": len(common), "examples": common[:10]}
    return result


def split_audit(train: pd.DataFrame, candidate: pd.DataFrame, kind: str) -> dict:
    if kind == "generator_disjoint":
        columns = CONTENT + ANCESTRY + HASHES
        allowed = ["source", "dataset"]
    elif kind == "source_disjoint":
        columns = ("source", "dataset") + CONTENT + ANCESTRY + HASHES
        allowed = list(GENERATORS)
    elif kind == "final":
        columns = ("source", "dataset") + CONTENT + ANCESTRY + HASHES + GENERATORS
        allowed = []
    else:
        raise ValueError(kind)
    result = overlaps(train, candidate, columns)
    if kind in {"generator_disjoint", "final"}:
        for component, column in (("voice", "voice_generator_family"),
                                  ("music", "music_generator_family")):
            left = train[train[f"{component}_fake"].eq(1)]
            right = candidate[candidate[f"{component}_fake"].eq(1)]
            common = sorted(tokens(left, column) & tokens(right, column))
            result[column] = {"count": len(common), "examples": common[:10]}
        columns = columns + GENERATORS
    return {
        "kind": kind,
        "rows": len(candidate),
        "rules": {"must_be_zero": list(columns), "allowed_overlap": allowed},
        "overlap": result,
        "pass": all(item["count"] == 0 for item in result.values()),
    }


def global_history_sources() -> tuple[set[str], list[str]]:
    sources: set[str] = set()
    scanned = []
    for path in sorted((ROOT / "data").glob("splits_v*/**/*.csv")):
        if path.parent == SPLITS or "final_holdout_v13b" in path.name:
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"source", "dataset"})
        except (ValueError, pd.errors.EmptyDataError):
            continue
        scanned.append(path.relative_to(ROOT).as_posix())
        for column in ("source", "dataset"):
            if column in frame:
                sources.update(tokens(frame, column))
    return sources, scanned


def main() -> None:
    train = pd.read_csv(SPLITS / "train.csv")
    generator_val = pd.read_csv(SPLITS / "val_generator_disjoint.csv")
    calibration = pd.read_csv(SPLITS / "cal_v13b.csv")
    source_val = read_optional("val_source_disjoint.csv")
    final = read_optional("final_holdout_v13b.csv")

    audits = {
        "generator_disjoint": split_audit(train, generator_val, "generator_disjoint"),
        "calibration": {
            "kind": "independent_calibration",
            "overlap": overlaps(train, calibration, CONTENT + ANCESTRY + HASHES),
        },
        "source_disjoint": (split_audit(train, source_val, "source_disjoint")
                            if source_val is not None else {"status": "NOT CREATED"}),
        "final_holdout": (split_audit(train, final, "final")
                          if final is not None else {"status": "NOT CREATED / NOT SEALED"}),
    }
    audits["calibration"]["pass"] = all(
        item["count"] == 0 for item in audits["calibration"]["overlap"].values())
    development_pass = (
        audits["generator_disjoint"]["pass"] and audits["calibration"]["pass"] and
        (source_val is None or audits["source_disjoint"]["pass"])
    )
    report = {
        "status": "PASS_DEVELOPMENT_SPLITS" if development_pass else "FAIL_LEAKAGE",
        "train_rows": len(train),
        "split_specific_rules": True,
        "audits": audits,
        "all_development_overlap_zero": development_pass,
        "final_holdout_metrics": "NOT READ / NOT RUN",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    historical_sources, scanned = global_history_sources()
    if final is None:
        history_report = {
            "status": "NOT CREATED / NOT SEALED",
            "metrics": "NOT READ / NOT RUN",
            "historical_manifests_scanned": len(scanned),
            "historical_sources": len(historical_sources),
        }
    else:
        final_sources = tokens(final, "source") | tokens(final, "dataset")
        common = sorted(final_sources & historical_sources)
        history_report = {
            "status": "PASS" if not common else "FAIL_HISTORY_OVERLAP",
            "metrics": "NOT READ / NOT RUN",
            "historical_manifests_scanned": len(scanned),
            "overlap_count": len(common), "overlap_examples": common[:10],
        }
    HISTORY_OUT.write_text(json.dumps(history_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
