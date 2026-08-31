#!/usr/bin/env python3
"""Calibration-only DF gate score/call frontier; never reads final/test data."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import df_execution_mask, robust_score


def adaptive_config(weights, enabled):
    return {
        "enabled": bool(enabled),
        "low": float(weights.get("adaptive_df_low", 0.2)),
        "high": float(weights.get("adaptive_df_high", 0.8)),
        "aggregation": str(weights.get("adaptive_df_aggregation", "max")),
        "minimum_duration": float(weights.get("adaptive_df_minimum_duration", 12.0)),
        "trigger_mode": str(weights.get("adaptive_df_trigger_mode", "primary")),
        "component_low": float(weights.get("adaptive_df_component_low", 0.3)),
        "component_high": float(weights.get("adaptive_df_component_high", 0.7)),
        "disagreement": float(weights.get("adaptive_df_disagreement", 0.3)),
    }


def second_crop_mask(cache, adaptive):
    if not adaptive["enabled"]:
        return np.zeros(len(cache), dtype=bool)
    primary = cache["df_primary"].to_numpy(float)
    mask = cache["duration_sec"].to_numpy(float) >= adaptive["minimum_duration"]
    mask &= (primary > adaptive["low"]) & (primary < adaptive["high"])
    mask &= np.isfinite(cache["df_second"].to_numpy(float))
    return mask


def policy_candidates():
    yield "all_df", {"df_gate_policy": "off"}
    yield "current_voice_0.8", {
        "df_gate_policy": "voice_presence", "df_gate_voice_presence_threshold": 0.8}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        yield f"voice_presence_{threshold}", {
            "df_gate_policy": "voice_presence",
            "df_gate_voice_presence_threshold": threshold}
        yield f"any_presence_{threshold}", {
            "df_gate_policy": "any_presence",
            "df_gate_presence_threshold": threshold}
    for low, high in ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7)):
        yield f"uncertainty_{low}_{high}", {
            "df_gate_policy": "specialist_uncertainty",
            "df_gate_uncertainty_low": low, "df_gate_uncertainty_high": high}
        for threshold in (0.7, 0.8, 0.9):
            yield f"presence_{threshold}_or_uncertainty_{low}_{high}", {
                "df_gate_policy": "presence_or_uncertainty",
                "df_gate_presence_threshold": threshold,
                "df_gate_uncertainty_low": low, "df_gate_uncertainty_high": high}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="experiments/fusion_calibration_predictions_16k.csv")
    parser.add_argument("--weights", default="model/fusion_weights.json")
    parser.add_argument("--output", default="experiments/df_gate_frontier.json")
    args = parser.parse_args()
    cache = pd.read_csv(ROOT / args.cache)
    base = json.loads((ROOT / args.weights).read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for name, policy in policy_candidates():
        signature = json.dumps(policy, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        weights = dict(base)
        weights.update(policy)
        execute = df_execution_mask(cache, weights)
        for adaptive_enabled in (False, True):
            adaptive = adaptive_config(base, adaptive_enabled)
            objective, folds = robust_score(cache, weights, adaptive)
            seconds = execute & second_crop_mask(cache, adaptive)
            rows.append({
                "policy": name,
                "adaptive": adaptive_enabled,
                "robust_objective": objective,
                "df_primary_fraction": float(execute.mean()),
                "df_second_fraction": float(seconds.mean()),
                "df_crop_fraction": float(execute.mean() + seconds.mean()),
                "metrics_by_fold": folds,
                "configuration": policy,
            })
    rows.sort(key=lambda row: (-row["robust_objective"], row["df_crop_fraction"]))
    best_score = rows[0]["robust_objective"]
    frontier = []
    for maximum_loss in (0.0, 0.001, 0.0025, 0.005, 0.01):
        feasible = [row for row in rows if row["robust_objective"] >= best_score - maximum_loss]
        selected = min(feasible, key=lambda row: (row["df_crop_fraction"], -row["robust_objective"]))
        frontier.append({"maximum_objective_loss": maximum_loss, **selected})
    report = {
        "status": "MEASURED_CALIBRATION_ONLY",
        "samples": len(cache),
        "best_objective": best_score,
        "all_df_primary": next(row for row in rows if row["policy"] == "all_df" and not row["adaptive"]),
        "all_df_adaptive": next(row for row in rows if row["policy"] == "all_df" and row["adaptive"]),
        "current_gate": next(row for row in rows if row["policy"] == "current_voice_0.8" and row["adaptive"]),
        "top": rows[:15],
        "frontier": frontier,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
