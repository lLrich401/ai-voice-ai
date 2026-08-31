#!/usr/bin/env python3
"""One-shot evaluation of the already selected final candidate."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import collect, score_frame
import script as submission

REPORT = ROOT / "experiments/final_holdout_v6_report.json"
CACHE = ROOT / "experiments/final_holdout_v6_predictions.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if REPORT.exists() or CACHE.exists():
        raise RuntimeError(
            "v6 final holdout was already evaluated; refusing repeated model selection")
    weights_path = ROOT / "model/fusion_weights.json"
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    if (weights.get("df_gate_policy") != "off"
            or bool(weights.get("adaptive_df_enabled"))
            or weights.get("voice_fake_aggregation") != "max"):
        raise RuntimeError("Final policy is not the preselected gate-off/primary/voice-max candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (submission.load_voice_model(device), submission.load_music_model(device))
    panns = submission.load_panns(device)
    df_session = submission.load_df_arena(device)
    split_path = ROOT / "data/splits/final_holdout.csv"
    frame = pd.read_csv(split_path)
    started = time.perf_counter()
    cache = pd.DataFrame(collect(
        "final_holdout", frame, models, df_session, panns, device, 16,
        df_config={"enabled": False, "force_second_for_long": False},
    ))
    elapsed = time.perf_counter() - started
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    metrics = score_frame(cache, weights, adaptive)
    cache.to_csv(CACHE, index=False)
    report = {
        "status": "MEASURED_ONE_SHOT_FINAL_LOCAL_HOLDOUT",
        "selection_locked_before_evaluation": True,
        "samples": len(cache),
        "metrics": metrics,
        "runtime_seconds": elapsed,
        "seconds_per_file": elapsed / len(cache),
        "df_primary_fraction": float(cache["df_used"].mean()),
        "split_sha256": sha256(split_path),
        "fusion_sha256": sha256(weights_path),
        "pipeline_version": submission.PIPELINE_VERSION,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
