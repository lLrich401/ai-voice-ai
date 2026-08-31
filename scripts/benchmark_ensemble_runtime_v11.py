#!/usr/bin/env python3
"""Measure incremental specialist cost with shared decode/segments on non-final VAL-A."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_ensemble_cache_v11 import load_model
from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden
import script as submission


def run(model, groups, device):
    began = time.perf_counter()
    submission._run_torch_segments(model, groups, device, use_amp=False)
    return time.perf_counter() - began


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="experiments/v11/runtime.json")
    args = parser.parse_args()
    output = ROOT / args.output
    split_path = ROOT / "data/splits/val_a.csv"
    assert_final_holdout_forbidden(output, split_path)
    device = torch.device(args.device)
    frame = pd.read_csv(split_path).sample(args.samples, random_state=20260930).reset_index(drop=True)
    decode_started = time.perf_counter()
    waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
             for _, row in frame.iterrows()]
    groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
    decode_seconds = time.perf_counter() - decode_started
    paths = {
        "v7_voice": (ROOT / "model/best.pt", "voice"),
        "v7_music": (ROOT / "model/music_best.pt", "music"),
        "v9_voice": (ROOT / "model/candidates/voice_aasist_v9.pt", "voice"),
        "v9_music": (ROOT / "model/candidates/music_spec_cnn_v9.pt", "music"),
    }
    models = {name: load_model(path, task, device) for name, (path, task) in paths.items()}
    timings = {name: [] for name in models}
    # Warm each model once, then use medians to reduce ordering/jit noise.
    for name, model in models.items():
        run(model, groups[:min(4, len(groups))], device)
        for _ in range(args.repeats):
            timings[name].append(run(model, groups, device))
    medians = {name: float(statistics.median(values)) for name, values in timings.items()}
    baseline_report = json.loads((ROOT / "experiments/v7/runtime.json").read_text(encoding="utf-8"))
    baseline_minutes = float(baseline_report["results"]["batch_8"]["projected_1200_minutes"])
    additions = {
        "baseline_v7": (), "music_ensemble": ("v9_music",),
        "voice_ensemble": ("v9_voice",), "joint_ensemble": ("v9_music", "v9_voice"),
    }
    projections = {}
    for name, extra in additions.items():
        extra_minutes = sum(medians[item] for item in extra) / args.samples * 1200.0 / 60.0
        projections[name] = {
            "projected_1200_minutes": baseline_minutes + extra_minutes,
            "incremental_minutes": extra_minutes,
            "under_60_minutes": baseline_minutes + extra_minutes < 60.0,
            "under_35_minutes": baseline_minutes + extra_minutes < 35.0,
        }
    report = {
        "status": "MEASURED_INCREMENTAL_CPU_PLUS_EXISTING_BASELINE_PROJECTION",
        "final_holdout": "NOT RUN", "device": str(device),
        "samples": args.samples, "repeats": args.repeats,
        "shared_decode_and_segment_seconds": decode_seconds,
        "model_run_seconds": timings, "model_median_seconds": medians,
        "baseline_source": "experiments/v7/runtime.json batch_8",
        "projections": projections,
        "note": "Projection adds measured CPU specialist overhead to the historical full-pipeline baseline; it is not official DACON server runtime.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
