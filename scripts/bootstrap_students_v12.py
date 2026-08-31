#!/usr/bin/env python3
"""Paired 1000-bootstrap stability analysis for every V12 student candidate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_students_v12 import VALS, replace_student
from src.ensemble import assert_final_holdout_forbidden, score_head_selective_ensemble
from src.metrics import compute_eer


def distribution(values: list[float], higher_is_better: bool = True) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "win_rate": float(np.mean(array > 0.0) if higher_is_better else np.mean(array < 0.0)),
    }


def sample_frame(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return frame.iloc[rng.integers(0, len(frame), len(frame))].reset_index(drop=True)


def run_candidate(task: str, candidate: str, iterations: int,
                  weights: dict, seed: int) -> dict:
    base_domains = {name: pd.read_csv(
        ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv") for name in VALS}
    candidate_domains = {name: replace_student(frame, task, ROOT /
        f"experiments/v12/cache/{task}_students/{candidate.lower()}/{name}.csv")
        for name, frame in base_domains.items()}
    base_expanded = pd.read_csv(
        ROOT / "experiments/v12/cache/v7_canonical/expanded_unseen.csv")
    candidate_expanded = replace_student(base_expanded, task, ROOT /
        f"experiments/v12/cache/{task}_students/{candidate.lower()}/expanded_unseen.csv")
    base_cal = {name: pd.read_csv(
        ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv")
        for name in ("cal_old", "cal_v12")}
    candidate_cal = {name: replace_student(frame, task, ROOT /
        f"experiments/v12/cache/{task}_students/{candidate.lower()}/{name}.csv")
        for name, frame in base_cal.items()}
    deltas = {name: [] for name in (
        "file_eer", "voice_eer", "music_eer", "ads", "total", "robust")}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        base_metrics, candidate_metrics = [], []
        for name in VALS:
            indices = rng.integers(0, len(base_domains[name]), len(base_domains[name]))
            base_sample = base_domains[name].iloc[indices].reset_index(drop=True)
            candidate_sample = candidate_domains[name].iloc[indices].reset_index(drop=True)
            base_metrics.append(score_head_selective_ensemble(base_sample, weights))
            candidate_metrics.append(score_head_selective_ensemble(candidate_sample, weights))
        for metric in ("file_eer", "voice_eer", "music_eer", "ads", "total"):
            deltas[metric].append(float(np.mean([value[metric] for value in candidate_metrics])
                                        - np.mean([value[metric] for value in base_metrics])))
        base_totals = [value["total"] for value in base_metrics]
        candidate_totals = [value["total"] for value in candidate_metrics]

        # Reuse paired indices for the two expanded sets.
        unseen_values = {}
        for component in ("voice", "music"):
            role = ("val_b_unseen_generator" if component == "voice"
                    else "val_b_unseen_music_generator")
            b = base_expanded[base_expanded["data_role"].astype(str).eq(role)].reset_index(drop=True)
            c = candidate_expanded[candidate_expanded["data_role"].astype(str).eq(role)].reset_index(drop=True)
            indices = rng.integers(0, len(b), len(b))
            column = "vf" if component == "voice" else "mf"
            truth = b.iloc[indices][f"y_{component}_fake"].to_numpy()
            unseen_values[f"base_{component}"] = compute_eer(
                truth, b.iloc[indices][column].to_numpy())
            unseen_values[f"candidate_{component}"] = compute_eer(
                truth, c.iloc[indices][column].to_numpy())

        cal_values = {}
        for cal_name in ("cal_old", "cal_v12"):
            base_fold_totals, candidate_fold_totals = [], []
            for fold_name, bfold in base_cal[cal_name].groupby("calibration_fold", sort=True):
                cfold = candidate_cal[cal_name][
                    candidate_cal[cal_name]["calibration_fold"].eq(fold_name)]
                indices = rng.integers(0, len(bfold), len(bfold))
                base_fold_totals.append(score_head_selective_ensemble(
                    bfold.iloc[indices], weights)["total"])
                candidate_fold_totals.append(score_head_selective_ensemble(
                    cfold.iloc[indices], weights)["total"])
            cal_values[f"base_{cal_name}"] = 0.7 * np.mean(base_fold_totals) + 0.3 * np.min(base_fold_totals)
            cal_values[f"candidate_{cal_name}"] = (
                0.7 * np.mean(candidate_fold_totals) + 0.3 * np.min(candidate_fold_totals))
        base_robust = (0.25 * np.mean(base_totals) + 0.15 * np.min(base_totals)
                       + 0.15 * (1 - unseen_values["base_voice"])
                       + 0.15 * (1 - unseen_values["base_music"])
                       + 0.15 * cal_values["base_cal_old"]
                       + 0.15 * cal_values["base_cal_v12"])
        candidate_robust = (0.25 * np.mean(candidate_totals) + 0.15 * np.min(candidate_totals)
                            + 0.15 * (1 - unseen_values["candidate_voice"])
                            + 0.15 * (1 - unseen_values["candidate_music"])
                            + 0.15 * cal_values["candidate_cal_old"]
                            + 0.15 * cal_values["candidate_cal_v12"])
        deltas["robust"].append(float(candidate_robust - base_robust))
    return {
        "candidate": candidate, "task": task, "iterations": iterations,
        "delta_candidate_minus_v7": {
            metric: distribution(values, higher_is_better=metric not in {
                "file_eer", "voice_eer", "music_eer"})
            for metric, values in deltas.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    output = ROOT / "experiments/v12/student_bootstrap.json"
    assert_final_holdout_forbidden(output)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    results = []
    for task, candidates in (("music", ["M1", "M2", "M3", "M4", "M5"]),
                             ("voice", ["V1", "V2", "V3", "V4", "V5"])):
        for index, candidate in enumerate(candidates):
            result = run_candidate(task, candidate, args.iterations, weights,
                                   20260901 + 100 * (task == "voice") + index)
            results.append(result)
            print(candidate, result["delta_candidate_minus_v7"]["robust"], flush=True)
    report = {"status": "MEASURED_PAIRED_BOOTSTRAP", "results": results,
              "final_holdout": "NOT RUN"}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
