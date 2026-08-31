#!/usr/bin/env python3
"""Compare voice segment aggregation on non-final validation domains."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import balanced_subset, score_frame
from src.dataset import load_manifest_row_wave
from src.metrics import compute_eer
import script as submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_split", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiments/voice_aggregation_report.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = submission.load_voice_model(device)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    no_adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    methods = ("mean", "max", "topk_mean", "logit_topk_mean")
    report = {"status": "MEASURED_NON_FINAL_LOCAL_VALIDATION", "splits": {}}
    for split_index, split in enumerate(("val_a", "val_b", "val_c", "val_d")):
        source = pd.read_csv(ROOT / "data/splits" / f"{split}.csv")
        frame = balanced_subset(source, args.per_split, 20260901 + split_index)
        predictions = {method: [] for method in methods}
        labels = []
        paths = []
        sources = []
        generators = []
        started = time.perf_counter()
        for start in range(0, len(frame), args.batch_size):
            rows = frame.iloc[start:start + args.batch_size]
            waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                     for _, row in rows.iterrows()]
            groups = [submission.limit_aux_segments(
                submission.select_aux_segments(wave, sr=16000, seg_sec=4.0), 3)
                for wave in waves]
            outputs, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
            for row_index, (_, row) in enumerate(rows.iterrows()):
                lower, upper = bounds[row_index]
                values = outputs["voice_fake"][lower:upper]
                labels.append((int(row["voice_fake"]), int(row["voice_present"])))
                paths.append(str(row["path"]))
                sources.append(str(row.get("source", "unknown")))
                generators.append(str(row.get("generator", "unknown")))
                for method in methods:
                    predictions[method].append(
                        submission.aggregate_predictions(values, method=method, top_k=2))
        present = [index for index, (_, exists) in enumerate(labels) if exists == 1]
        y = [labels[index][0] for index in present]
        metrics = {method: {
            "voice_eer": compute_eer(y, [predictions[method][index] for index in present])}
            for method in methods}
        aggregation_frame = pd.DataFrame({
            "path": paths,
            "source": sources,
            "generator": generators,
            "voice_fake": [value for value, _ in labels],
            "voice_present": [value for _, value in labels],
            **{f"vf_{method}": values for method, values in predictions.items()},
        })
        aggregation_frame.to_csv(
            ROOT / "experiments" / f"{split}_voice_aggregation.csv", index=False)
        fusion_metrics = {}
        feature_path = ROOT / "experiments" / f"{split}_features_16k.csv"
        if feature_path.exists():
            features = pd.read_csv(feature_path)
            for left, right, name in (
                (features["source"].astype(str), aggregation_frame["source"].astype(str), "source"),
                (features["generator"].astype(str), aggregation_frame["generator"].astype(str), "generator"),
                (features["y_voice_fake"].astype(int), aggregation_frame["voice_fake"].astype(int), "voice_fake"),
                (features["y_voice_present"].astype(int), aggregation_frame["voice_present"].astype(int), "voice_present"),
            ):
                if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
                    raise RuntimeError(f"{split} feature-cache ordering mismatch: {name}")
            for method in methods:
                trial = features.copy()
                trial["vf"] = aggregation_frame[f"vf_{method}"].to_numpy()
                fusion_metrics[method] = score_frame(trial, weights, no_adaptive)
        report["splits"][split] = {
            "samples": len(frame), "voice_present_samples": len(present),
            "runtime_seconds": time.perf_counter() - started,
            "methods": metrics,
            "fusion_metrics": fusion_metrics,
        }
    current = "topk_mean"
    candidates = []
    for method in methods:
        deltas = {
            split: payload["methods"][method]["voice_eer"]
                   - payload["methods"][current]["voice_eer"]
            for split, payload in report["splits"].items()}
        fused_deltas = {
            split: payload["fusion_metrics"][method]["total"]
                   - payload["fusion_metrics"][current]["total"]
            for split, payload in report["splits"].items()
            if payload["fusion_metrics"]}
        candidates.append({"method": method, "mean_eer": sum(
            payload["methods"][method]["voice_eer"]
            for payload in report["splits"].values()) / 4.0,
            "delta_vs_current": deltas,
            "no_domain_regression": all(value <= 1e-12 for value in deltas.values()),
            "fusion_total_delta_vs_current": fused_deltas,
            "fusion_no_domain_regression": all(value >= -1e-12 for value in fused_deltas.values())})
    report["comparison"] = candidates
    report["decision"] = "ADOPT only if both VOICE EER and fused TOTAL have no domain regression"
    output = ROOT / args.output
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
