#!/usr/bin/env python3
"""Measure candidate fusion/gate policies on non-final validation domains."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import balanced_subset, collect, score_frame
import script as submission


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adaptive_from_weights(weights, enabled):
    return {
        "enabled": enabled,
        "low": weights.get("adaptive_df_low", 0.2),
        "high": weights.get("adaptive_df_high", 0.8),
        "aggregation": weights.get("adaptive_df_aggregation", "max"),
        "minimum_duration": weights.get("adaptive_df_minimum_duration", 12.0),
        "trigger_mode": weights.get("adaptive_df_trigger_mode", "primary"),
        "component_low": weights.get("adaptive_df_component_low", 0.3),
        "component_high": weights.get("adaptive_df_component_high", 0.7),
        "disagreement": weights.get("adaptive_df_disagreement", 0.3),
    }


def policies(base):
    choices = {
        "A_current_voice_0.8": {
            "df_gate_policy": "voice_presence", "df_gate_voice_presence_threshold": 0.8},
        "B_full_primary": {"df_gate_policy": "off"},
        "C_full_adaptive": {"df_gate_policy": "off"},
        "D_voice_0.7_adaptive": {
            "df_gate_policy": "voice_presence", "df_gate_voice_presence_threshold": 0.7},
        "E_any_presence_0.8_adaptive": {
            "df_gate_policy": "any_presence", "df_gate_presence_threshold": 0.8},
        "F_presence_or_uncertainty": {
            "df_gate_policy": "presence_or_uncertainty",
            "df_gate_presence_threshold": 0.8,
            "df_gate_uncertainty_low": 0.2,
            "df_gate_uncertainty_high": 0.8},
    }
    for name, override in choices.items():
        weights = dict(base)
        weights.update(override)
        adaptive = name not in ("A_current_voice_0.8", "B_full_primary")
        yield name, weights, adaptive_from_weights(base, adaptive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val_a", "val_b", "val_c", "val_d"])
    parser.add_argument("--per_split", type=int, default=128, help="0 evaluates every row")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiments/validation_domain_gate_report.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    models = (submission.load_voice_model(device), submission.load_music_model(device))
    panns = submission.load_panns(device)
    df_session = submission.load_df_arena(device)
    report = {
        "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
        "per_split_limit": args.per_split,
        "splits": {},
    }
    for index, split in enumerate(args.splits):
        split_path = ROOT / "data/splits" / f"{split}.csv"
        original = pd.read_csv(split_path)
        frame = balanced_subset(original, args.per_split, 20260901 + index)
        started = time.perf_counter()
        cache = pd.DataFrame(collect(
            split, frame, models, df_session, panns, device, args.batch_size))
        elapsed = time.perf_counter() - started
        cache_path = ROOT / "experiments" / f"{split}_features_16k.csv"
        cache.to_csv(cache_path, index=False)
        results = {}
        for name, policy_weights, adaptive in policies(weights):
            metrics = score_frame(cache, policy_weights, adaptive)
            if policy_weights["df_gate_policy"] == "off":
                fraction = 1.0
            elif policy_weights["df_gate_policy"] == "voice_presence":
                fraction = float((cache["vp_model"] >= policy_weights["df_gate_voice_presence_threshold"]).mean())
            elif policy_weights["df_gate_policy"] == "any_presence":
                fraction = float((np.maximum(cache["vp_model"], cache["mp_model"])
                                  >= policy_weights["df_gate_presence_threshold"]).mean())
            else:
                presence = np.maximum(cache["vp_model"], cache["mp_model"])
                specialist = 0.5 * cache["vfile"] + 0.5 * cache["mfile"]
                fraction = float(((presence >= policy_weights["df_gate_presence_threshold"])
                                  | specialist.between(
                                      policy_weights["df_gate_uncertainty_low"],
                                      policy_weights["df_gate_uncertainty_high"],
                                      inclusive="neither")).mean())
            results[name] = {"metrics": metrics, "df_primary_fraction": fraction}
        report["splits"][split] = {
            "samples": len(cache), "source_rows": len(original),
            "split_sha256": sha256(split_path),
            "runtime_seconds_feature_collection": elapsed,
            "policies": results,
        }
        (ROOT / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({split: report["splits"][split]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
