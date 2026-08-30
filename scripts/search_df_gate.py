#!/usr/bin/env python3
"""Search calibration-only DF-Arena triage policies and report speed/score trade-offs."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_dacon_metrics
import script as submission

HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def score(cache, weights, gate):
    metrics = {}
    for fold, frame in cache.groupby("calibration_fold"):
        predictions = []
        for enabled, row in zip(gate.loc[frame.index], frame.itertuples()):
            row_weights = dict(weights)
            df_score = row.df_primary
            if not enabled:
                row_weights["w_df_voice_component"] = 0.0
                row_weights["w_df_music_component"] = 0.0
                df_score = 0.5
            predictions.append(submission.fuse_prediction_features(
                df_score, row.vf, row.mf, row.vfile, row.mfile, row.vp_model,
                row.mp_model, row.vp_panns, row.mp_panns, row_weights))
        predictions = np.asarray(predictions)
        y_true = {head: frame[f"y_{head}"].to_numpy() for head in HEADS}
        y_pred = {head: predictions[:, index] for index, head in enumerate(HEADS)}
        metrics[fold] = compute_dacon_metrics(y_true, y_pred)
    scores = np.asarray([item["score"] for item in metrics.values()])
    return 0.7 * scores.mean() + 0.3 * scores.min(), metrics


def main():
    cache = pd.read_csv(ROOT / "experiments/fusion_calibration_predictions.csv")
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    candidates = []
    for low, high in ((0.05, 0.95), (0.10, 0.90), (0.20, 0.80),
                      (0.25, 0.75), (0.30, 0.70), (0.40, 0.60)):
        masks = {
            "voice_uncertain": cache["vf"].between(low, high, inclusive="neither"),
            "voice_or_file_uncertain": (cache["vf"].between(low, high, inclusive="neither")
                                         | cache["vfile"].between(low, high, inclusive="neither")),
            "any_component_uncertain": (cache["vf"].between(low, high, inclusive="neither")
                                        | cache["mf"].between(low, high, inclusive="neither")),
        }
        for name, mask in masks.items():
            objective, metrics = score(cache, weights, mask)
            candidates.append({"policy": name, "low": low, "high": high,
                               "df_fraction": float(mask.mean()),
                               "objective": float(objective), "metrics": metrics})
    for column in ("vp_model", "vp_panns"):
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            mask = cache[column] >= threshold
            objective, metrics = score(cache, weights, mask)
            candidates.append({"policy": f"{column}_at_least", "threshold": threshold,
                               "df_fraction": float(mask.mean()),
                               "objective": float(objective), "metrics": metrics})
    candidates.sort(key=lambda item: item["objective"], reverse=True)
    frontier = []
    for limit in (0.25, 0.50, 0.60, 0.70, 0.80, 1.0):
        feasible = [item for item in candidates if item["df_fraction"] <= limit]
        if feasible:
            frontier.append({"maximum_df_fraction": limit, **feasible[0]})
    print(json.dumps({"top": candidates[:10], "frontier": frontier}, indent=2))


if __name__ == "__main__":
    main()
