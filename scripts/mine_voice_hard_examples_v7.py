#!/usr/bin/env python3
"""Score TRAIN only for leakage-safe hard-negative/positive sampling."""
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

import script as submission
from src.dataset import load_manifest_row_wave


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", default="experiments/v7/train_hard_scores.csv")
    args = parser.parse_args()
    split_path = ROOT / "data/splits/train.csv"
    frame = pd.read_csv(split_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = submission.load_voice_model(device)
    records = []
    started = time.perf_counter()
    for lower in range(0, len(frame), args.batch_size):
        rows = frame.iloc[lower:lower + args.batch_size]
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                 for _, row in rows.iterrows()]
        groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
        output, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
        for (_, row), (first, last) in zip(rows.iterrows(), bounds):
            score = submission.aggregate_predictions(output["voice_fake"][first:last], "max", 2)
            records.append({
                "path": str(row["path"]), "data_role": "train",
                "voice_fake": int(row["voice_fake"]),
                "voice_present": int(row["voice_present"]),
                "source": str(row.get("source", "unknown")),
                "generator": str(row.get("generator", "unknown")),
                "voice_fake_score": score,
            })
    scores = pd.DataFrame(records)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output, index=False)
    voice = scores[scores["voice_present"].eq(1)].copy()
    voice["hardness"] = np.where(
        voice["voice_fake"].eq(1), 1.0 - voice["voice_fake_score"], voice["voice_fake_score"])
    report = {
        "status": "MEASURED_TRAIN_ONLY",
        "final_holdout": "NOT USED",
        "rows": int(len(scores)), "voice_rows": int(len(voice)),
        "split_sha256": sha256(split_path),
        "checkpoint_sha256": sha256(ROOT / "model/best.pt"),
        "runtime_seconds": time.perf_counter() - started,
        "hardest_real": voice[voice["voice_fake"].eq(0)].nlargest(20, "hardness")[
            ["path", "source", "generator", "voice_fake_score"]].to_dict("records"),
        "hardest_fake": voice[voice["voice_fake"].eq(1)].nlargest(20, "hardness")[
            ["path", "source", "generator", "voice_fake_score"]].to_dict("records"),
    }
    (output.parent / "hard_negative_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "rows", "voice_rows", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
