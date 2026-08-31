#!/usr/bin/env python3
"""Recompute calibration voice aggregation without rerunning DF/PANNs."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import balanced_subset, cache_metadata
from src.dataset import load_manifest_row_wave
import script as submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="experiments/fusion_calibration_predictions_16k.csv")
    parser.add_argument("--output", default="experiments/fusion_calibration_predictions_16k_voice_max.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline_ref", default="1b5553200d08dcf4f7867e7ecfc8cc93a5d62d5f")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = submission.load_voice_model(device)
    split_path = ROOT / "data/splits/fusion_calibration.csv"
    frame = balanced_subset(pd.read_csv(split_path), 0, 20260831)
    scores = []
    for start in range(0, len(frame), args.batch_size):
        rows = frame.iloc[start:start + args.batch_size]
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                 for _, row in rows.iterrows()]
        groups = [submission.limit_aux_segments(
            submission.select_aux_segments(wave, sr=16000, seg_sec=4.0), 3)
            for wave in waves]
        output, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
        for lower, upper in bounds:
            scores.append(submission.aggregate_predictions(
                output["voice_fake"][lower:upper], method="max", top_k=2))
    base = pd.read_csv(ROOT / args.base)
    for left, right, name in (
        (base["source"].astype(str), frame["source"].astype(str), "source"),
        (base["generator"].astype(str), frame["generator"].astype(str), "generator"),
        (base["y_voice_fake"].astype(int), frame["voice_fake"].astype(int), "voice_fake"),
        (base["y_voice_present"].astype(int), frame["voice_present"].astype(int), "voice_present"),
    ):
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            raise RuntimeError(f"Voice aggregation ordering mismatch: {name}")
    base["vf"] = scores
    output_path = ROOT / args.output
    base.to_csv(output_path, index=False)
    metadata = cache_metadata([split_path])
    metadata["baseline_ref"] = args.baseline_ref
    metadata["voice_fake_aggregation"] = "max"
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(base)} max-aggregated calibration rows to {output_path}")


if __name__ == "__main__":
    main()
