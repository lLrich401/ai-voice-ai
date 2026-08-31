#!/usr/bin/env python3
"""VOICE segment selection/aggregation A/B on VAL-A/B/C/D only."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script as submission
from scripts.calibrate_fusion import score_frame
from src.dataset import load_manifest_row_wave
from src.metrics import compute_eer

SPLITS = ("val_a", "val_b", "val_c", "val_d")
POLICIES = ("high_energy", "uniform", "centered", "energy_diverse")
METHODS = ("mean", "max", "topk_mean", "logit_topk_mean", "logit_mean", "median",
           "trimmed_mean", "max_mean_0.25", "max_mean_0.5", "max_mean_0.75")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", default="experiments/v7/voice_segment_policy.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = submission.load_voice_model(device)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    results = []
    sample_records = []
    started = time.perf_counter()
    for split in SPLITS:
        aggregation = pd.read_csv(ROOT / "experiments" / f"{split}_voice_aggregation.csv")
        source = pd.read_csv(ROOT / "data/splits" / f"{split}.csv")
        if source["path"].astype(str).duplicated().any():
            raise RuntimeError(f"{split}: duplicate path")
        source = source.set_index(source["path"].astype(str), drop=False)
        rows = source.loc[aggregation["path"].astype(str)].reset_index(drop=True)
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                 for _, row in rows.iterrows()]
        base_features = pd.read_csv(ROOT / "experiments" / f"{split}_features_16k.csv")
        high_energy_frame = None
        for policy in POLICIES:
            groups = [submission.select_aux_segments(wave, policy=policy) for wave in waves]
            outputs = {name: [] for name in ("voice_fake", "file_fake", "voice_present")}
            segment_voice_scores = []
            for lower in range(0, len(groups), args.batch_size):
                batch_groups = groups[lower:lower + args.batch_size]
                batch_out, bounds = submission._run_torch_segments(model, batch_groups, device, use_amp=True)
                for first, last in bounds:
                    values = batch_out["voice_fake"][first:last]
                    segment_voice_scores.append([float(value) for value in values])
                    outputs["voice_fake"].append(values)
                    outputs["file_fake"].append(batch_out["file_fake"][first:last])
                    outputs["voice_present"].append(batch_out["voice_present"][first:last])
            for method in METHODS:
                trial = base_features.copy()
                trial["vf"] = [submission.aggregate_predictions(values, method, 2)
                               for values in outputs["voice_fake"]]
                trial["vfile"] = [submission.aggregate_predictions(values, "topk_mean", 2)
                                  for values in outputs["file_fake"]]
                trial["vp_model"] = [float(np.mean(values)) for values in outputs["voice_present"]]
                present = trial["y_voice_present"].to_numpy(dtype=int) == 1
                raw_eer = compute_eer(
                    trial.loc[present, "y_voice_fake"], trial.loc[present, "vf"])
                fused = score_frame(trial, weights, adaptive)
                results.append({
                    "split": split, "selection": policy, "aggregation": method,
                    "raw_voice_eer": raw_eer, "voice_eer": fused["voice_eer"],
                    "file_eer": fused["file_eer"], "music_eer": fused["music_eer"],
                    "voice_auc": fused["voice_auc"], "music_auc": fused["music_auc"],
                    "ads": fused["ads"], "cps": fused["cps"], "total": fused["total"],
                })
                if policy == "high_energy" and method == "max":
                    high_energy_frame = pd.DataFrame({
                        "path": rows["path"].astype(str),
                        "vf_max": trial["vf"].astype(float),
                        "vfile": trial["vfile"].astype(float),
                        "vp_model": trial["vp_model"].astype(float),
                    })
            if policy == "high_energy":
                for index, scores in enumerate(segment_voice_scores):
                    sample_records.append({
                        "split": split, "path": str(rows.iloc[index]["path"]),
                        "voice_fake": int(rows.iloc[index]["voice_fake"]),
                        "voice_present": int(rows.iloc[index]["voice_present"]),
                        "segment_count": len(scores),
                        "segment_scores": json.dumps(scores),
                        "segment_spread": max(scores) - min(scores),
                    })
        if high_energy_frame is None:
            raise RuntimeError(f"{split}: high-energy/max feature capture failed")
        high_energy_frame.to_csv(
            ROOT / "experiments/v7" / f"{split}_voice_features.csv", index=False)
    table = pd.DataFrame(results)
    summary = []
    for (selection, aggregation), group in table.groupby(["selection", "aggregation"]):
        voice_quality = 1.0 - group["voice_eer"].to_numpy(float)
        summary.append({
            "selection": selection, "aggregation": aggregation,
            "mean_voice_eer": float(group["voice_eer"].mean()),
            "worst_voice_eer": float(group["voice_eer"].max()),
            "mean_total": float(group["total"].mean()),
            "worst_total": float(group["total"].min()),
            "voice_robust_objective": float(0.6 * voice_quality.mean() + 0.4 * voice_quality.min()),
        })
    baseline = table[(table["selection"] == "high_energy") & (table["aggregation"] == "max")].set_index("split")
    for candidate in summary:
        current = table[(table["selection"] == candidate["selection"])
                        & (table["aggregation"] == candidate["aggregation"])].set_index("split")
        candidate["voice_no_domain_regression"] = bool(
            (current["voice_eer"] <= baseline["voice_eer"] + 1e-12).all())
        candidate["total_no_domain_regression"] = bool(
            (current["total"] >= baseline["total"] - 1e-12).all())
        candidate["val_d_voice_eer"] = float(current.loc["val_d", "voice_eer"])
    baseline_summary = next(row for row in summary
                            if row["selection"] == "high_energy" and row["aggregation"] == "max")
    eligible = [row for row in summary
                if row["voice_no_domain_regression"] and row["total_no_domain_regression"]
                and row["mean_voice_eer"] < baseline_summary["mean_voice_eer"] - 1e-12]
    selected = (max(eligible, key=lambda row: (row["voice_robust_objective"], row["worst_total"]))
                if eligible else baseline_summary)
    selected["decision"] = ("ADOPT" if eligible else
                             "KEEP_BASELINE_NO_STRICT_VOICE_EER_IMPROVEMENT")
    payload = {
        "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
        "final_holdout": "NOT RUN",
        "runtime_seconds": time.perf_counter() - started,
        "baseline": {"selection": "high_energy", "aggregation": "max"},
        "selection_rule": "strict mean VOICE EER gain, then robust objective, subject to no VAL-A/B/C/D VOICE or TOTAL regression",
        "selected": selected,
        "domain_results": results,
        "summary": summary,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(sample_records).to_csv(output.parent / "voice_segment_scores.csv", index=False)
    print(json.dumps({"selected": selected, "runtime_seconds": payload["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
