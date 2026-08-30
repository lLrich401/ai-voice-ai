#!/usr/bin/env python3
"""Measure canonical inference throughput without inspecting holdout labels."""
import argparse
import json
import pathlib
import sys
import time

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
import script as submission


def run_batches(models, df_session, panns, waves, device, batch_size, config,
                specialist_max_segments=3, panns_max_segments=3):
    start = time.perf_counter()
    df_calls = 0
    for index in range(0, len(waves), batch_size):
        features = submission.infer_wave_features_batch(
            *models, df_session, panns, waves[index:index + batch_size], device,
            use_demucs=False, df_config=config,
            specialist_max_segments=specialist_max_segments,
            panns_max_segments=panns_max_segments)
        df_calls += sum(bool(feature.get("df_used", True)) for feature in features)
    return time.perf_counter() - start, df_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    frame = pd.read_csv(ROOT / "data/splits/final_holdout.csv")
    frame = frame[~frame["path"].astype(str).str.startswith("MIX::")]
    frame = frame.sample(min(args.samples, len(frame)), random_state=20260831)
    waves = [load_manifest_row_wave(row, is_training=False) for _, row in frame.iterrows()]
    models = (submission.load_voice_model(device), submission.load_music_model(device))
    panns = submission.load_panns(device)
    df_session = submission.load_df_arena(device)
    # Warm all kernels/models before timing.
    submission.infer_wave_features_batch(
        *models, df_session, panns, waves[:1], device, use_demucs=False,
        df_config={"enabled": False})
    results = {}
    for name, config, specialist_segments, panns_segments in (
        ("baseline_single_crop", {"enabled": False}, 3, 3),
        ("adaptive_crop", {"enabled": True, "low": 0.2, "high": 0.8,
                           "aggregation": "max"}, 3, 3),
        ("fast_single_crop", {"enabled": False}, 2, 1),
        ("gated_df", {"enabled": False, "gate_voice_presence_threshold": 0.8}, 3, 3),
        ("no_df", {"enabled": False, "gate_voice_presence_threshold": 2.0}, 3, 3),
    ):
        elapsed, df_calls = run_batches(models, df_session, panns, waves, device, args.batch_size, config,
                                        specialist_segments, panns_segments)
        results[name] = {"samples": len(waves), "seconds": elapsed,
                         "df_calls": df_calls,
                         "seconds_per_file": elapsed / len(waves),
                         "projected_1200_minutes": elapsed / len(waves) * 1200 / 60}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
