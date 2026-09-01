#!/usr/bin/env python3
"""Recompute the fail-closed V13B stage state from checked-in evidence."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED = {"PASS", "PASS_WITH_WARNING", "FAIL", "NOT_RUN"}
DEVELOPMENT_BASE_COMMIT = "e8434b9c368ee5de3368d8b0b04559cf19c3ffaa"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_optional(relative: str) -> dict | None:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def evaluate_gates(dataset: dict, shortcut: dict, policy: dict) -> dict:
    exploratory_policy = policy["exploratory_training_gates"]
    effective = lambda value: max(float(value), 1.0 - float(value))
    exploratory_checks = {
        "shortcut_status_pass": shortcut["status"] == "PASS",
        "metadata_auc": effective(shortcut["metadata_only"]["auc"]) <= exploratory_policy["metadata_auc_max"],
        "acoustic_auc": effective(shortcut["acoustic_only"]["auc"]) <= exploratory_policy["acoustic_auc_max"],
        "combined_auc": effective(shortcut["combined"]["auc"]) <= exploratory_policy["combined_auc_hard_max"],
        "split_identifier_overlap_zero": bool(dataset["structural_gates"]["split_identifier_overlap_zero"]),
        "paired_voice_sources": dataset["paired_voice_sources"] >= exploratory_policy["paired_voice_sources_min"],
        "paired_music_sources": dataset["paired_music_sources"] >= exploratory_policy["paired_music_sources_min"],
        "voice_fake_generators": len(dataset["voice_fake_generators_train"]) >= exploratory_policy["voice_fake_generators_min"],
        "music_fake_generators": len(dataset["music_fake_generators_train"]) >= exploratory_policy["music_fake_generators_min"],
        "partial_data": bool(dataset["partial_rows"] > 0),
        "balanced_rr_rf_fr_ff": set(dataset["mixed_state_counts"]) == {"RR", "RF", "FR", "FF"}
            and len(set(dataset["mixed_state_counts"].values())) == 1,
    }
    adoption_policy = policy["adoption_data_gates"]
    adoption_checks = {
        "exploratory_training_gate": all(exploratory_checks.values()),
        "paired_music_sources": dataset["paired_music_sources"] >= adoption_policy["paired_music_sources_min"],
        "approved_metric_complete_source_disjoint_validation": bool(
            dataset["structural_gates"]["approved_metric_complete_source_disjoint_validation"]),
        "sealed_unused_final_holdout": bool(dataset["structural_gates"]["final_holdout_v13b_sealed"]),
    }
    return {
        "exploratory_checks": exploratory_checks,
        "exploratory_allowed": all(exploratory_checks.values()),
        "adoption_checks": adoption_checks,
        "adoption_eligible": all(adoption_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/v13b/stage_status.json")
    args = parser.parse_args()
    latest = load("experiments/latest_results.json")
    dataset = load("data/splits_v13b/DATASET_V13B.json")
    shortcut = load("experiments/v13b/source_shortcut_audit.json")
    policy = load("configs/v13b/selection_policy.json")
    music_evaluation = load_optional("experiments/v13b/music_exploratory_evaluation.json")
    file_evaluation = load_optional("experiments/v13b/file_exploratory_evaluation.json")
    runtime_evaluation = load_optional("experiments/v13b/end_to_end_runtime_benchmark.json")
    m1_evaluation = load_optional("experiments/v13b/m1_music_representation_probe.json")
    m2_evaluation = load_optional("experiments/v13b/m2_music_representation_probe.json")
    file_complementarity = load_optional("experiments/v13b/m2_file_complementarity.json")
    gates = evaluate_gates(dataset, shortcut, policy)
    failed = [name for name, value in gates["adoption_checks"].items() if not value]
    stages = {
        "0_baseline_freeze": "PASS",
        "1_latest_results_fix": "PASS" if latest["selected_submission"]["name"] == "TEST5" else "FAIL",
        "2_dataset_redesign": "PASS" if gates["adoption_eligible"] else "PASS_WITH_WARNING",
        "3_shortcut_audit": "PASS" if shortcut["status"] == "PASS" else "FAIL",
        "4_validation_source_acquisition": "PASS" if gates["adoption_eligible"] else "FAIL",
    }
    for stage in ("5_music_architecture", "6_file_architecture", "7_voice_architecture",
                  "8_scaled_training", "9_robust_validation", "10_calibration", "11_runtime",
                  "12_bootstrap", "13_candidate_lock", "14_final_holdout", "15_submit_build"):
        stages[stage] = "NOT_RUN"
    if music_evaluation:
        stages["5_music_architecture"] = (
            "PASS" if music_evaluation["comparison"]["exploratory_improved"] else "PASS_WITH_WARNING")
    if file_evaluation:
        stages["6_file_architecture"] = (
            "PASS" if file_evaluation["comparison"]["exploratory_improved"] else "PASS_WITH_WARNING")
    if music_evaluation or file_evaluation:
        stages["9_robust_validation"] = "PASS_WITH_WARNING"
    if m1_evaluation or m2_evaluation:
        stages["5_music_architecture"] = "PASS_WITH_WARNING"
        stages["9_robust_validation"] = "PASS_WITH_WARNING"
    if file_complementarity:
        stages["6_file_architecture"] = "PASS_WITH_WARNING"
    if runtime_evaluation:
        stages["11_runtime"] = "PASS" if runtime_evaluation["status"] == "PASS" else "FAIL"
    if not set(stages.values()) <= ALLOWED:
        raise RuntimeError("stage manager generated an unsupported state")
    report = {
        "version": "V13B_STAGE_MANAGER_20260901",
        "branch": git_value("branch", "--show-current"),
        "development_base_commit": DEVELOPMENT_BASE_COMMIT,
        "current_git_commit": git_value("rev-parse", "HEAD"),
        "decision": "KEEP_TEST5",
        "current_stage": (4 if m1_evaluation and m2_evaluation else
                          7 if music_evaluation and file_evaluation else
                          6 if music_evaluation else 5 if gates["exploratory_allowed"] else 4),
        "stages": stages,
        "failed_stage_blocks_later_stages": False,
        "exploratory_training": {
            "status": "ALLOWED_NOT_SELECTED" if gates["exploratory_allowed"] else "BLOCKED",
            "checks": gates["exploratory_checks"],
            "permitted_stages": [5, 6, 7, 8, 9] if gates["exploratory_allowed"] else [],
        },
        "adoption": {
            "status": "ELIGIBLE" if gates["adoption_eligible"] else "BLOCKED_BY_DATA_GATES",
            "checks": gates["adoption_checks"],
            "blocked_stages": [] if gates["adoption_eligible"] else [10, 12, 13, 14, 15],
        },
        "model_training": ("ALLOWED_EXPLORATORY_NOT_ADOPTABLE"
                           if gates["exploratory_allowed"] else "BLOCKED"),
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
        "adoption_blocking_gates": failed,
        "selected_artifacts_mutated": False,
        "historical_v13_preserved": bool(dataset["historical_v13_preserved"]),
        "final_holdout_v13b_read_or_scored": False,
    }
    current_commit = report["current_git_commit"]
    latest["branch"] = report["branch"]
    latest["development_base_commit"] = DEVELOPMENT_BASE_COMMIT
    latest["current_git_commit"] = current_commit
    latest.pop("development_git_commit", None)
    latest["development"]["current_stage"] = report["current_stage"]
    latest["development"]["current_stage_name"] = (
        "Music representation screening complete; second Music/source-disjoint acquisition blocked"
        if m1_evaluation and m2_evaluation else
        "exploratory model research allowed; adoption data blocked"
        if gates["exploratory_allowed"] and not gates["adoption_eligible"]
        else "adoption eligible" if gates["adoption_eligible"] else "exploratory gate blocked")
    latest["development"]["exploratory_training"] = report["exploratory_training"]["status"]
    latest["development"]["adoption"] = report["adoption"]["status"]
    latest["development"]["v13b_exploratory_results"] = {
        "music": ({
            "status": music_evaluation["status"],
            "baseline_music_eer": music_evaluation["comparison"]["music_eer_baseline"],
            "candidate_music_eer": music_evaluation["comparison"]["music_eer_candidate"],
            "selected": False,
        } if music_evaluation else "NOT RUN"),
        "file": ({
            "status": file_evaluation["status"],
            "baseline_standalone_file_eer": file_evaluation["comparison"]["file_eer_baseline"],
            "candidate_standalone_file_eer": file_evaluation["comparison"]["file_eer_candidate"],
            "selected": False,
        } if file_evaluation else "NOT RUN"),
        "source_disjoint": "NOT MEASURED",
        "final_holdout": "NOT RUN",
    }
    latest["development"]["v13b_representation_results"] = {
        "m1_music_eer": (m1_evaluation["generator_disjoint"]["music_eer"]
                         if m1_evaluation else "NOT RUN"),
        "m2_music_eer": (m2_evaluation["generator_disjoint"]["music_eer"]
                         if m2_evaluation else "NOT RUN"),
        "m2_file_complementarity": ({
            "canonical_file_eer": file_complementarity["canonical_test5_file_eer"],
            "candidate_file_eer": file_complementarity["m2_candidate_file_eer"],
            "best_same_split_fusion_file_eer": file_complementarity[
                "best_same_split_exploratory_fusion"]["file_eer"],
            "adoptable": False,
        } if file_complementarity else "NOT RUN"),
    }
    latest["not_run"] = [
        "V13B Voice exploratory candidate",
        "V13B source-disjoint robust validation",
        "V13B bootstrap",
        "FINAL_HOLDOUT_V13B evaluation",
        "new DACON submission",
    ]
    (ROOT / "experiments/latest_results.json").write_text(
        json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
