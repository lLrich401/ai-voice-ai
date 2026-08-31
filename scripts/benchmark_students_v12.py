#!/usr/bin/env python3
"""Measured local CPU/XPU runtime for V7 and student-only V12 specialists."""

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

from scripts.build_canonical_cache_v12 import selected_rows
from scripts.train_distilled_v12 import load_checkpoint_model
from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden
import script as submission


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-candidate", default="V2")
    parser.add_argument("--music-candidate", default="M4")
    args = parser.parse_args()
    voice_candidate = args.voice_candidate.upper()
    music_candidate = args.music_candidate.upper()
    output = ROOT / "experiments/v12/runtime.json"
    assert_final_holdout_forbidden(output)
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    _, rows, _ = selected_rows("val_a")
    rows = rows.iloc[:64]
    began = time.perf_counter()
    waves = [load_manifest_row_wave(
        row, sr=16000, is_training=False, use_demucs=False) for _, row in rows.iterrows()]
    groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
    decode_seconds = time.perf_counter() - began
    paths = {
        "v7_voice": ROOT / "model/best.pt",
        f"v12_{voice_candidate.lower()}_voice": (
            ROOT / f"model/candidates/v12/{voice_candidate.lower()}_student.pt"),
        "v7_music": ROOT / "model/music_best.pt",
        f"v12_{music_candidate.lower()}_music": (
            ROOT / f"model/candidates/v12/{music_candidate.lower()}_student.pt"),
    }
    models = {name: load_checkpoint_model(path, device)[0].eval() for name, path in paths.items()}
    timings = {name: [] for name in models}
    with torch.no_grad():
        for name, model in models.items():
            submission._run_torch_segments(model, groups[:2], device, use_amp=True)
            synchronize(device)
            for _ in range(3):
                began = time.perf_counter()
                submission._run_torch_segments(model, groups, device, use_amp=True)
                synchronize(device)
                timings[name].append(time.perf_counter() - began)
    baseline_projection = json.loads((ROOT / "experiments/v7/runtime.json").read_text(
        encoding="utf-8"))["results"]["batch_8"]["projected_1200_minutes"]
    candidate_key = f"v12_{voice_candidate.lower()}_voice"
    voice_delta = ((statistics.median(timings[candidate_key])
                    - statistics.median(timings["v7_voice"])) / 64 * 1200 / 60)
    music_key = f"v12_{music_candidate.lower()}_music"
    music_delta = ((statistics.median(timings[music_key])
                    - statistics.median(timings["v7_music"])) / 64 * 1200 / 60)
    report = {
        "status": "MEASURED_LOCAL_SPECIALIST_PLUS_EXISTING_FULL_PIPELINE_PROJECTION",
        "device": str(device), "files": 64, "repeats": 3,
        "shared_decode_seconds": decode_seconds,
        "timings_seconds": timings,
        "median_seconds": {name: statistics.median(values) for name, values in timings.items()},
        "projected_1200_minutes": {
            "v7": baseline_projection,
            f"v12_{music_candidate.lower()}_{voice_candidate.lower()}_student_only": (
                baseline_projection + voice_delta + music_delta),
        },
        "model_count": {
            "v7": 4,
            f"v12_{music_candidate.lower()}_{voice_candidate.lower()}_student_only": 4,
        },
        "official_server_runtime": "NOT RUN",
        "final_holdout": "NOT RUN",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
