#!/usr/bin/env python3
"""Paired bootstrap for the M4 music + V2 voice V12 joint candidate."""

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
from scripts.bootstrap_students_v12 import distribution
from src.ensemble import assert_final_holdout_forbidden, score_head_selective_ensemble
from src.metrics import compute_eer


def joint(base: pd.DataFrame, name: str) -> pd.DataFrame:
    result = replace_student(
        base, "music", ROOT / f"experiments/v12/cache/music_students/m4/{name}.csv")
    return replace_student(
        result, "voice", ROOT / f"experiments/v12/cache/voice_students/v2/{name}.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    output = ROOT / "experiments/v12/joint_bootstrap.json"
    assert_final_holdout_forbidden(output)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    base = {name: pd.read_csv(
        ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv")
        for name in (*VALS, "expanded_unseen", "cal_old", "cal_v12")}
    candidate = {name: joint(frame, name) for name, frame in base.items()}
    deltas = {name: [] for name in (
        "file_eer", "voice_eer", "music_eer", "ads", "total", "robust")}
    rng = np.random.default_rng(20260912)
    for _ in range(args.iterations):
        base_metrics, candidate_metrics = [], []
        for name in VALS:
            indices = rng.integers(0, len(base[name]), len(base[name]))
            base_metrics.append(score_head_selective_ensemble(base[name].iloc[indices], weights))
            candidate_metrics.append(score_head_selective_ensemble(
                candidate[name].iloc[indices], weights))
        for metric in ("file_eer", "voice_eer", "music_eer", "ads", "total"):
            deltas[metric].append(float(
                np.mean([value[metric] for value in candidate_metrics])
                - np.mean([value[metric] for value in base_metrics])))
        unseen = {}
        for component, role, column in (
            ("voice", "val_b_unseen_generator", "vf"),
            ("music", "val_b_unseen_music_generator", "mf"),
        ):
            b = base["expanded_unseen"][
                base["expanded_unseen"]["data_role"].astype(str).eq(role)].reset_index(drop=True)
            c = candidate["expanded_unseen"][
                candidate["expanded_unseen"]["data_role"].astype(str).eq(role)].reset_index(drop=True)
            indices = rng.integers(0, len(b), len(b))
            truth = b.iloc[indices][f"y_{component}_fake"].to_numpy()
            unseen[f"base_{component}"] = compute_eer(truth, b.iloc[indices][column].to_numpy())
            unseen[f"candidate_{component}"] = compute_eer(
                truth, c.iloc[indices][column].to_numpy())
        cal = {}
        for cal_name in ("cal_old", "cal_v12"):
            bt, ct = [], []
            for fold_name, bfold in base[cal_name].groupby("calibration_fold", sort=True):
                cfold = candidate[cal_name][
                    candidate[cal_name]["calibration_fold"].eq(fold_name)]
                indices = rng.integers(0, len(bfold), len(bfold))
                bt.append(score_head_selective_ensemble(bfold.iloc[indices], weights)["total"])
                ct.append(score_head_selective_ensemble(cfold.iloc[indices], weights)["total"])
            cal[f"base_{cal_name}"] = 0.7 * np.mean(bt) + 0.3 * np.min(bt)
            cal[f"candidate_{cal_name}"] = 0.7 * np.mean(ct) + 0.3 * np.min(ct)
        base_totals = [value["total"] for value in base_metrics]
        candidate_totals = [value["total"] for value in candidate_metrics]
        base_robust = (0.25 * np.mean(base_totals) + 0.15 * np.min(base_totals)
                       + 0.15 * (1 - unseen["base_voice"])
                       + 0.15 * (1 - unseen["base_music"])
                       + 0.15 * cal["base_cal_old"] + 0.15 * cal["base_cal_v12"])
        candidate_robust = (0.25 * np.mean(candidate_totals) + 0.15 * np.min(candidate_totals)
                            + 0.15 * (1 - unseen["candidate_voice"])
                            + 0.15 * (1 - unseen["candidate_music"])
                            + 0.15 * cal["candidate_cal_old"]
                            + 0.15 * cal["candidate_cal_v12"])
        deltas["robust"].append(float(candidate_robust - base_robust))
    report = {
        "candidate": "M4_music_plus_V2_voice",
        "iterations": args.iterations,
        "delta_candidate_minus_v7": {
            metric: distribution(values, higher_is_better=metric not in {
                "file_eer", "voice_eer", "music_eer"})
            for metric, values in deltas.items()
        },
        "final_holdout": "NOT RUN",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
