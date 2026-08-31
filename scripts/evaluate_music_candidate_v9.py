#!/usr/bin/env python3
"""Strictly compare the v9 music candidate on non-final validation domains."""

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
from scripts.calibrate_fusion import score_frame
from src.dataset import load_manifest_row_wave
from src.metrics import compute_eer
from src.models.beats_backbone import MusicMultitask

SPLITS = ("val_a", "val_b", "val_c", "val_d")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_music(path: pathlib.Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("task") != "music" or checkpoint.get("backbone") != "spec_cnn":
        raise RuntimeError("not a strict music SpecCNN checkpoint")
    model = MusicMultitask(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def evaluate(name, path, device, weights, adaptive):
    model, metadata = load_music(path, device)
    started = time.perf_counter()
    domains = {}
    for split in SPLITS:
        aggregation = pd.read_csv(ROOT / "experiments" / f"{split}_voice_aggregation.csv")
        source = pd.read_csv(ROOT / "data/splits" / f"{split}.csv")
        source = source.set_index(source["path"].astype(str), drop=False)
        rows = source.loc[aggregation["path"].astype(str)].reset_index(drop=True)
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False,
                                        use_demucs=False, task="music")
                 for _, row in rows.iterrows()]
        groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
        output, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
        music = [submission.aggregate_predictions(output["music_fake"][a:b], "topk_mean", 2)
                 for a, b in bounds]
        file_score = [submission.aggregate_predictions(output["file_fake"][a:b], "topk_mean", 2)
                      for a, b in bounds]
        presence = [float(np.mean(output["music_present"][a:b])) for a, b in bounds]
        features = pd.read_csv(ROOT / "experiments" / f"{split}_features_16k.csv")
        features["mf"], features["mfile"], features["mp_model"] = music, file_score, presence
        present = features["y_music_present"].to_numpy(dtype=int) == 1
        metrics = score_frame(features, weights, adaptive)
        domains[split] = {
            "raw_music_eer": compute_eer(
                features.loc[present, "y_music_fake"], np.asarray(music)[present]),
            "music_eer": metrics["music_eer"], "file_eer": metrics["file_eer"],
            "total": metrics["total"],
        }
    return {
        "candidate": name, "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
        "checkpoint_sha256": sha256(path), "checkpoint_metadata_score": metadata.get("score"),
        "domains": domains,
        "mean_music_eer": float(np.mean([v["music_eer"] for v in domains.values()])),
        "worst_music_eer": float(np.max([v["music_eer"] for v in domains.values()])),
        "mean_total": float(np.mean([v["total"] for v in domains.values()])),
        "worst_total": float(np.min([v["total"] for v in domains.values()])),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="model/candidates/music_spec_cnn_v9.pt")
    parser.add_argument("--output", default="experiments/v9/music_architecture_ablation.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    results = [
        evaluate("current_music_spec_cnn", ROOT / "model/music_best.pt", device, weights, adaptive),
        evaluate("v9_expanded_music_spec_cnn", ROOT / args.candidate, device, weights, adaptive),
    ]
    baseline, candidate = results
    candidate["no_domain_music_regression"] = all(
        candidate["domains"][split]["music_eer"]
        <= baseline["domains"][split]["music_eer"] + 1e-12 for split in SPLITS
    )
    candidate["no_domain_total_regression"] = all(
        candidate["domains"][split]["total"]
        >= baseline["domains"][split]["total"] - 1e-12 for split in SPLITS
    )
    adopt = (candidate["no_domain_music_regression"]
             and candidate["no_domain_total_regression"]
             and candidate["mean_music_eer"] < baseline["mean_music_eer"] - 1e-12)
    payload = {
        "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION", "final_holdout": "NOT RUN",
        "results": results, "selected": candidate["candidate"] if adopt else baseline["candidate"],
        "decision": "ADOPT" if adopt else "KEEP_CURRENT_NO_STRICT_ROBUST_IMPROVEMENT",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
