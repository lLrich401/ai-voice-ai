#!/usr/bin/env python3
"""Evaluate V12 student checkpoints on frozen canonical caches."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ensemble import assert_final_holdout_forbidden, score_head_selective_ensemble
from src.metrics import compute_eer


VALS = ("val_a", "val_b", "val_c", "val_d")


def robust_fold_summary(frame: pd.DataFrame, weights: dict) -> dict:
    folds = {str(name): score_head_selective_ensemble(group, weights)
             for name, group in frame.groupby("calibration_fold", sort=True)}
    totals = np.asarray([metrics["total"] for metrics in folds.values()], dtype=float)
    return {"folds": folds, "mean": float(totals.mean()),
            "worst": float(totals.min()),
            "robust": float(0.7 * totals.mean() + 0.3 * totals.min())}


def replace_student(frame: pd.DataFrame, task: str, cache: pathlib.Path) -> pd.DataFrame:
    student = pd.read_csv(cache)
    if student["path"].astype(str).tolist() != frame["path"].astype(str).tolist():
        raise RuntimeError(f"student cache alignment failure: {cache}")
    result = frame.copy()
    if task == "music":
        result["mf"] = student["component_fake"]
        result["mfile"] = student["file_fake"]
        result["mp_model"] = student["component_present"]
    else:
        result["vf"] = student["component_fake"]
        result["vfile"] = student["file_fake"]
        result["vp_model"] = student["component_present"]
    return result


def unseen_metrics(frame: pd.DataFrame, weights: dict) -> dict:
    del weights
    result = {}
    for task, role, column in (
        ("voice", "val_b_unseen_generator", "vf"),
        ("music", "val_b_unseen_music_generator", "mf"),
    ):
        mask = frame["data_role"].astype(str).eq(role).to_numpy()
        result[f"{task}_eer"] = float(compute_eer(
            frame.loc[mask, f"y_{task}_fake"].to_numpy(),
            frame.loc[mask, column].to_numpy()))
    return result


def summarize(candidate: str, task: str, weights: dict) -> dict:
    domains = {}
    for split in VALS:
        base_path = ROOT / f"experiments/v12/cache/v7_canonical/{split}.csv"
        frame = pd.read_csv(base_path)
        if candidate != "BASELINE_V7":
            frame = replace_student(frame, task, ROOT /
                f"experiments/v12/cache/{task}_students/{candidate.lower()}/{split}.csv")
        domains[split] = score_head_selective_ensemble(frame, weights)
    expanded = pd.read_csv(ROOT / "experiments/v12/cache/v7_canonical/expanded_unseen.csv")
    if candidate != "BASELINE_V7":
        expanded = replace_student(expanded, task, ROOT /
            f"experiments/v12/cache/{task}_students/{candidate.lower()}/expanded_unseen.csv")
    unseen = unseen_metrics(expanded, weights)
    calibrations = {}
    for name in ("cal_old", "cal_v12"):
        frame = pd.read_csv(ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv")
        if candidate != "BASELINE_V7":
            frame = replace_student(frame, task, ROOT /
                f"experiments/v12/cache/{task}_students/{candidate.lower()}/{name}.csv")
        calibrations[name] = robust_fold_summary(frame, weights)
    totals = np.asarray([metrics["total"] for metrics in domains.values()])
    mean_metrics = {key: float(np.mean([metrics[key] for metrics in domains.values()]))
                    for key in ("file_eer", "voice_eer", "music_eer", "ads", "cps", "total")}
    robust = (0.25 * float(totals.mean()) + 0.15 * float(totals.min())
              + 0.15 * (1.0 - unseen["voice_eer"])
              + 0.15 * (1.0 - unseen["music_eer"])
              + 0.15 * calibrations["cal_old"]["robust"]
              + 0.15 * calibrations["cal_v12"]["robust"])
    return {"candidate": candidate, "domains": domains, "mean": mean_metrics,
            "worst_total": float(totals.min()), "unseen": unseen,
            "calibration": calibrations, "robust_objective": float(robust)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["music", "voice"], required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    output = ROOT / f"experiments/v12/{args.task}_student_evaluation.json"
    assert_final_holdout_forbidden(output)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    results = [summarize("BASELINE_V7", args.task, weights)]
    results.extend(summarize(candidate.upper(), args.task, weights)
                   for candidate in args.candidates)
    baseline = results[0]
    gates = json.loads((ROOT / "experiments/v12/selection_policy_precommitted.json").read_text(
        encoding="utf-8"))["hard_gates"]
    for result in results:
        result["delta_robust"] = result["robust_objective"] - baseline["robust_objective"]
        result["hard_gate"] = {
            "file": result["mean"]["file_eer"] - baseline["mean"]["file_eer"]
                    <= gates["maximum_file_eer_regression"],
            "voice_unseen": result["unseen"]["voice_eer"] - baseline["unseen"]["voice_eer"]
                    <= gates["maximum_voice_unseen_eer_regression"],
            "music_unseen": result["unseen"]["music_eer"] - baseline["unseen"]["music_eer"]
                    <= gates["maximum_music_unseen_eer_regression"],
            "worst_total": baseline["worst_total"] - result["worst_total"]
                    <= gates["maximum_worst_total_regression"],
        }
        result["hard_gate_pass"] = all(result["hard_gate"].values())
    eligible = [result for result in results[1:]
                if result["hard_gate_pass"] and result["delta_robust"] > 0]
    selected = max(eligible, key=lambda result: result["robust_objective"]) if eligible else None
    report = {
        "status": "MEASURED_NON_FINAL",
        "task": args.task, "objective": (
            "0.25 mean VAL TOTAL + 0.15 worst VAL TOTAL + 0.15 voice unseen quality + "
            "0.15 music unseen quality + 0.15 CAL_OLD robust + 0.15 CAL_V12 robust"),
        "results": results,
        "point_estimate_selected": selected["candidate"] if selected else "KEEP_V7",
        "bootstrap_required_before_adoption": True,
        "final_holdout": "NOT RUN",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = pd.DataFrame([{
        "candidate": result["candidate"], **result["mean"],
        "worst_total": result["worst_total"],
        "voice_unseen_eer": result["unseen"]["voice_eer"],
        "music_unseen_eer": result["unseen"]["music_eer"],
        "cal_old": result["calibration"]["cal_old"]["robust"],
        "cal_v12": result["calibration"]["cal_v12"]["robust"],
        "robust_objective": result["robust_objective"],
        "delta_robust": result["delta_robust"],
        "hard_gate_pass": result["hard_gate_pass"],
    } for result in results])
    summary.to_csv(output.with_suffix(".csv"), index=False)
    print(summary.to_string(index=False))
    print("point_estimate_selected", report["point_estimate_selected"])


if __name__ == "__main__":
    main()
