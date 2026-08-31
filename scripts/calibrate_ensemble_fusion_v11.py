#!/usr/bin/env python3
"""Constrained calibration of the selected joint ensemble; no final holdout."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_ensemble_calibration_v11 import fold_summary
from scripts.search_head_ensemble_v11 import (
    DOMAINS, bootstrap, config_predictions, evaluate_config, load_cache,
)
from src.ensemble import assert_final_holdout_forbidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", default="experiments/v11/ensemble_search.json")
    parser.add_argument("--cache_dir", default="experiments/v11/cache")
    parser.add_argument("--output", default="experiments/v11/recalibrated_fusion.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    paths = [ROOT / args.search, ROOT / args.cache_dir, ROOT / args.output]
    assert_final_holdout_forbidden(*paths)
    search = json.loads(paths[0].read_text(encoding="utf-8"))
    config = search["best_joint"]["config"]
    baseline_config = search["baseline"]["config"]
    initial = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    calibration = load_cache(paths[1] / "fusion_calibration.csv",
                             ROOT / "data/splits/fusion_calibration.csv")
    unseen = load_cache(paths[1] / "expanded_unseen.csv",
                        ROOT / "data/splits_v9_candidate/val_b.csv")
    validation = {name: load_cache(paths[1] / f"original_{name}.csv",
                                   ROOT / f"data/splits/{name}.csv") for name in DOMAINS}

    candidates = []
    for voice_df, music_df, file_df in itertools.product(
        (0.20, 0.25, 0.30, 0.35), (0.0, 0.05, 0.10), (0.20, 0.25, 0.30)):
        weights = dict(initial)
        weights.update({
            "w_df_voice_component": voice_df,
            "w_df_music_component": music_df,
            "w_df_arena": file_df,
        })
        predicted = config_predictions(calibration, weights, config)
        summary = fold_summary(calibration, predicted)
        candidates.append({
            "weights": {key: weights[key] for key in (
                "w_df_voice_component", "w_df_music_component", "w_df_arena")},
            "calibration": summary,
        })
    selected = max(candidates, key=lambda item: item["calibration"]["robust_objective"])
    selected_weights = dict(initial)
    selected_weights.update(selected["weights"])
    baseline_validation = evaluate_config(validation, unseen, initial, baseline_config)
    candidate_validation = evaluate_config(validation, unseen, selected_weights, config)
    bootstrap_report = bootstrap(
        validation, selected_weights, baseline_config, config, args.bootstrap, 20260920)
    # The bootstrap helper uses one weight set for both configurations. Also
    # report the exact initial-v7 comparison separately below.
    rng = np.random.default_rng(20260921)
    exact_robust_delta = []
    base_predictions = {name: config_predictions(frame, initial, baseline_config)
                        for name, frame in validation.items()}
    candidate_predictions = {name: config_predictions(frame, selected_weights, config)
                             for name, frame in validation.items()}
    from scripts.search_head_ensemble_v11 import metric_from_predictions, robust_summary, distribution
    for _ in range(args.bootstrap):
        base_metrics, candidate_metrics = {}, {}
        for name, frame in validation.items():
            index = rng.integers(0, len(frame), len(frame))
            sampled = frame.iloc[index].reset_index(drop=True)
            base_metrics[name] = metric_from_predictions(sampled, base_predictions[name][index])
            candidate_metrics[name] = metric_from_predictions(sampled, candidate_predictions[name][index])
        exact_robust_delta.append(
            robust_summary(candidate_metrics)["robust_objective"]
            - robust_summary(base_metrics)["robust_objective"])
    baseline_calibration = fold_summary(
        calibration, config_predictions(calibration, initial, baseline_config))
    adoption = bool(
        selected["calibration"]["robust_objective"] > baseline_calibration["robust_objective"]
        and candidate_validation["robust_objective"] > baseline_validation["robust_objective"]
        and np.mean(np.asarray(exact_robust_delta) > 0.0) >= 0.6)
    report = {
        "status": "MEASURED_CONSTRAINED_CALIBRATION",
        "final_holdout": "NOT RUN", "grid_size": len(candidates),
        "baseline_calibration": baseline_calibration,
        "selected_calibration": selected,
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "bootstrap_same_selected_weights": bootstrap_report,
        "bootstrap_exact_candidate_vs_v7": {
            "iterations": args.bootstrap,
            "robust_delta": distribution(exact_robust_delta),
        },
        "adopt": adoption,
        "candidate_f_logistic_meta": (
            "ELIGIBLE_NOT_RUN" if adoption else
            "NOT RUN: candidate E failed stable-improvement gate"),
    }
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    paths[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_weights": selected["weights"],
        "calibration_delta": (selected["calibration"]["robust_objective"]
                              - baseline_calibration["robust_objective"]),
        "validation_delta": (candidate_validation["robust_objective"]
                             - baseline_validation["robust_objective"]),
        "bootstrap": report["bootstrap_exact_candidate_vs_v7"],
        "adopt": adoption, "candidate_f": report["candidate_f_logistic_meta"],
    }, indent=2))


if __name__ == "__main__":
    main()
