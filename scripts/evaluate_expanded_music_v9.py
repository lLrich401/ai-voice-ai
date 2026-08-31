#!/usr/bin/env python3
"""Compare music checkpoints on expanded, non-final VAL-A/B/C/D."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.beats_backbone import MusicMultitask
from src.train import validate_multisegment


def load(path: pathlib.Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("task") != "music" or checkpoint.get("backbone") != "spec_cnn":
        raise RuntimeError(f"not a strict music SpecCNN checkpoint: {path}")
    model = MusicMultitask(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="data/splits_v9_candidate")
    parser.add_argument("--candidate", default="model/candidates/music_spec_cnn_v9.pt")
    parser.add_argument("--output", default="experiments/v9/music_expanded_domain_results.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    paths = {
        "current_music_spec_cnn": pathlib.Path("model/music_best.pt"),
        "v9_expanded_music_spec_cnn": pathlib.Path(args.candidate),
    }
    results = {}
    for name, path in paths.items():
        model = load(path, device)
        domains = {}
        for split in ("val_a", "val_b", "val_c", "val_d"):
            frame = pd.read_csv(pathlib.Path(args.splits) / f"{split}.csv")
            metrics = validate_multisegment(
                model, frame, device, use_demucs=False, task="music", batch_size=32
            )
            domains[split] = {key: float(metrics[key]) for key in (
                "music_eer", "music_auc", "file_eer", "score"
            )}
        results[name] = {"checkpoint_sha256": sha256(path), "domains": domains}
    baseline = results["current_music_spec_cnn"]["domains"]
    candidate = results["v9_expanded_music_spec_cnn"]["domains"]
    no_eer_regression = all(candidate[s]["music_eer"] <= baseline[s]["music_eer"] + 1e-12
                            for s in baseline)
    no_auc_regression = all(candidate[s]["music_auc"] >= baseline[s]["music_auc"] - 1e-12
                            for s in baseline)
    payload = {
        "status": "MEASURED_NON_FINAL_EXPANDED_VALIDATION", "final_holdout": "NOT RUN",
        "results": results, "no_domain_music_eer_regression": no_eer_regression,
        "no_domain_music_auc_regression": no_auc_regression,
        "decision": "ELIGIBLE" if no_eer_regression and no_auc_regression else "REJECT",
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
