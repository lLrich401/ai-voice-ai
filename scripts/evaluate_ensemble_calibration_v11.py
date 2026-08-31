#!/usr/bin/env python3
"""One-shot independent calibration check for validation-selected ensembles."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.search_head_ensemble_v11 import (
    config_predictions, distribution, load_cache, metric_from_predictions,
)
from src.ensemble import assert_final_holdout_forbidden


def fold_summary(frame, predicted) -> dict:
    metrics = {}
    for fold in sorted(frame["calibration_fold"].astype(str).unique()):
        mask = frame["calibration_fold"].astype(str).eq(fold).to_numpy()
        metrics[fold] = metric_from_predictions(frame.loc[mask].reset_index(drop=True), predicted[mask])
    totals = np.asarray([value["total"] for value in metrics.values()])
    ads = np.asarray([value["ads"] for value in metrics.values()])
    return {
        "folds": metrics,
        "mean_total": float(totals.mean()), "worst_total": float(totals.min()),
        "mean_ads": float(ads.mean()),
        "robust_objective": float(0.7 * totals.mean() + 0.3 * totals.min()),
    }


def paired_bootstrap(frame, weights, baseline_config, candidate_config, iterations, seed):
    rng = np.random.default_rng(seed)
    base = config_predictions(frame, weights, baseline_config)
    candidate = config_predictions(frame, weights, candidate_config)
    folds = sorted(frame["calibration_fold"].astype(str).unique())
    fold_indices = {fold: np.flatnonzero(frame["calibration_fold"].astype(str).eq(fold))
                    for fold in folds}
    robust_delta, ads_delta, total_delta = [], [], []
    for _ in range(iterations):
        base_fold, candidate_fold = {}, {}
        for fold, indices in fold_indices.items():
            sampled_indices = rng.choice(indices, size=len(indices), replace=True)
            sampled = frame.iloc[sampled_indices].reset_index(drop=True)
            base_fold[fold] = metric_from_predictions(sampled, base[sampled_indices])
            candidate_fold[fold] = metric_from_predictions(sampled, candidate[sampled_indices])
        base_totals = np.asarray([base_fold[fold]["total"] for fold in folds])
        candidate_totals = np.asarray([candidate_fold[fold]["total"] for fold in folds])
        base_robust = 0.7 * base_totals.mean() + 0.3 * base_totals.min()
        candidate_robust = 0.7 * candidate_totals.mean() + 0.3 * candidate_totals.min()
        robust_delta.append(candidate_robust - base_robust)
        ads_delta.append(float(np.mean([
            candidate_fold[fold]["ads"] - base_fold[fold]["ads"] for fold in folds])))
        total_delta.append(float((candidate_totals - base_totals).mean()))
    return {
        "iterations": iterations,
        "robust_delta": distribution(robust_delta),
        "ads_delta": distribution(ads_delta),
        "total_delta": distribution(total_delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default="experiments/v11/ensemble_search.json")
    parser.add_argument("--cache", default="experiments/v11/cache/fusion_calibration.csv")
    parser.add_argument("--output", default="experiments/v11/calibration_check.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    paths = [ROOT / args.search, ROOT / args.cache, ROOT / args.output]
    assert_final_holdout_forbidden(*paths)
    search = json.loads(paths[0].read_text(encoding="utf-8"))
    frame = load_cache(paths[1], ROOT / "data/splits/fusion_calibration.csv")
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    configs = {
        "A_v7": search["baseline"]["config"],
        "B_music": search["best_music"]["config"],
        "C_voice": search["best_voice"]["config"],
        "D_joint": search["best_joint"]["config"],
    }
    predictions = {name: config_predictions(frame, weights, config)
                   for name, config in configs.items()}
    results = {name: fold_summary(frame, predicted) for name, predicted in predictions.items()}
    bootstrap = {name: paired_bootstrap(
        frame, weights, configs["A_v7"], config, args.bootstrap, 20260910 + index)
        for index, (name, config) in enumerate(configs.items()) if name != "A_v7"}
    baseline = results["A_v7"]
    for name, result in results.items():
        result["delta_robust_vs_v7"] = result["robust_objective"] - baseline["robust_objective"]
    report = {
        "status": "MEASURED_INDEPENDENT_CALIBRATION",
        "final_holdout": "NOT RUN", "results": results,
        "bootstrap": bootstrap,
        "adoption_gate": {
            name: bool(results[name]["robust_objective"] > baseline["robust_objective"]
                       and bootstrap[name]["robust_delta"]["win_rate"] >= 0.6)
            for name in bootstrap
        },
    }
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    paths[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
