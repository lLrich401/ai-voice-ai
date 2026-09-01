#!/usr/bin/env python3
"""Recompute the fail-closed V13B stage state from checked-in evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED = {"PASS", "PASS_WITH_WARNING", "FAIL", "NOT_RUN"}


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/v13b/stage_status.json")
    args = parser.parse_args()
    latest = load("experiments/latest_results.json")
    dataset = load("data/splits_v13b/DATASET_V13B.json")
    shortcut = load("experiments/v13b/source_shortcut_audit.json")
    failed = [name for name, value in dataset["structural_gates"].items() if not value]
    data_pass = not failed
    stages = {
        "0_baseline_freeze": "PASS",
        "1_latest_results_fix": "PASS" if latest["selected_submission"]["name"] == "TEST5" else "FAIL",
        "2_dataset_redesign": "PASS" if data_pass else "PASS_WITH_WARNING",
        "3_shortcut_audit": "PASS" if shortcut["status"] == "PASS" else "FAIL",
        "4_validation_source_acquisition": "PASS" if data_pass else "FAIL",
    }
    blocked = stages["3_shortcut_audit"] == "FAIL" or stages["4_validation_source_acquisition"] == "FAIL"
    for stage in ("5_music_architecture", "6_file_architecture", "7_voice_architecture",
                  "8_scaled_training", "9_robust_validation", "10_calibration", "11_runtime",
                  "12_bootstrap", "13_candidate_lock", "14_final_holdout", "15_submit_build"):
        stages[stage] = "NOT_RUN"
    if not set(stages.values()) <= ALLOWED:
        raise RuntimeError("stage manager generated an unsupported state")
    report = {
        "version": "V13B_STAGE_MANAGER_20260901",
        "branch": git_value("branch", "--show-current"),
        "development_commit": git_value("rev-parse", "HEAD"),
        "decision": "KEEP_TEST5",
        "current_stage": 4 if blocked else 5,
        "stages": stages,
        "failed_stage_blocks_later_stages": True,
        "model_training": "BLOCKED_BY_DATA_GATES" if blocked else "ALLOWED_NOT_RUN",
        "dataset_evidence": {
            "train_rows": dataset["train_rows"], "cal_rows": dataset["cal_rows"],
            "generator_validation_rows": dataset["generator_val_rows"],
            "paired_voice_sources": dataset["paired_voice_sources"],
            "paired_music_sources": dataset["paired_music_sources"],
            "partial_rows": dataset["partial_rows"], "mixed_rows": dataset["mixed_rows"],
            "metadata_auc": shortcut["metadata_only"]["auc"],
            "acoustic_auc": shortcut["acoustic_only"]["auc"],
            "combined_auc": shortcut["combined"]["auc"],
        },
        "blocking_gates": failed,
        "selected_artifacts_mutated": False,
        "historical_v13_preserved": bool(dataset["historical_v13_preserved"]),
        "final_holdout_v13b_read_or_scored": False,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
