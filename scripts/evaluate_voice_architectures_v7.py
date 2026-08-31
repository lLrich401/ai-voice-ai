#!/usr/bin/env python3
"""Strictly compare available VOICE checkpoints on non-final validation domains."""
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("backbone") != "spec_cnn" or checkpoint.get("task") != "voice":
        raise RuntimeError("not a strict voice SpecCNN checkpoint")
    model = MusicMultitask(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-candidate", action="append", default=[],
                        help="additional NAME=CHECKPOINT pair")
    parser.add_argument("--only-current-and-extra", action="store_true")
    parser.add_argument("--output", default="experiments/v7/voice_architecture_ablation.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    candidates = [
        ("current_spec_cnn", ROOT / "model/best.pt"),
        ("retained_legacy_spec_cnn", ROOT / "model/voice_spec_cnn.pt"),
        ("v7_retrain_baseline", ROOT / "model/candidates/v7_baseline_seed20260901.pt"),
        ("v7_channel_augmentation", ROOT / "model/candidates/v7_channel_seed20260830.pt"),
        ("v7_partial_fake", ROOT / "model/candidates/v7_partial_seed20260830.pt"),
        ("v7_hard_mining", ROOT / "model/candidates/v7_hard_seed20260830.pt"),
    ]
    extras = []
    for value in args.extra_candidate:
        if "=" not in value:
            raise ValueError("--extra-candidate must be NAME=CHECKPOINT")
        name, path = value.split("=", 1)
        extras.append((name, ROOT / path))
    if args.only_current_and_extra:
        candidates = candidates[:1]
    candidates.extend(extras)
    results = []
    for name, path in candidates:
        if not path.exists():
            results.append({"candidate": name, "status": "NOT RUN", "reason": "checkpoint missing"})
            continue
        try:
            model, metadata = load_spec(path, device)
        except Exception as error:
            results.append({"candidate": name, "status": "REJECTED", "reason": str(error)})
            continue
        started = time.perf_counter()
        domains = {}
        for split in SPLITS:
            aggregation = pd.read_csv(ROOT / "experiments" / f"{split}_voice_aggregation.csv")
            source = pd.read_csv(ROOT / "data/splits" / f"{split}.csv")
            source = source.set_index(source["path"].astype(str), drop=False)
            rows = source.loc[aggregation["path"].astype(str)].reset_index(drop=True)
            waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                     for _, row in rows.iterrows()]
            groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
            output, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
            voice = [submission.aggregate_predictions(output["voice_fake"][a:b], "max", 2) for a, b in bounds]
            file_score = [submission.aggregate_predictions(output["file_fake"][a:b], "topk_mean", 2) for a, b in bounds]
            presence = [float(np.mean(output["voice_present"][a:b])) for a, b in bounds]
            features = pd.read_csv(ROOT / "experiments" / f"{split}_features_16k.csv")
            features["vf"], features["vfile"], features["vp_model"] = voice, file_score, presence
            present = features["y_voice_present"].to_numpy(dtype=int) == 1
            metrics = score_frame(features, weights, adaptive)
            domains[split] = {
                "raw_voice_eer": compute_eer(
                    features.loc[present, "y_voice_fake"], np.asarray(voice)[present]),
                "voice_eer": metrics["voice_eer"], "file_eer": metrics["file_eer"],
                "total": metrics["total"],
            }
        eers = np.asarray([item["voice_eer"] for item in domains.values()])
        totals = np.asarray([item["total"] for item in domains.values()])
        results.append({
            "candidate": name, "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
            "checkpoint_sha256": sha256(path), "checkpoint_metadata_score": metadata.get("score"),
            "domains": domains, "mean_voice_eer": float(eers.mean()),
            "worst_voice_eer": float(eers.max()), "mean_total": float(totals.mean()),
            "worst_total": float(totals.min()), "runtime_seconds": time.perf_counter() - started,
        })
    measured = [item for item in results if item["status"].startswith("MEASURED")]
    baseline = next(item for item in measured if item["candidate"] == "current_spec_cnn")
    for item in measured:
        item["no_domain_voice_regression"] = all(
            item["domains"][split]["voice_eer"] <= baseline["domains"][split]["voice_eer"] + 1e-12
            for split in SPLITS)
        item["no_domain_total_regression"] = all(
            item["domains"][split]["total"] >= baseline["domains"][split]["total"] - 1e-12
            for split in SPLITS)
    eligible = [item for item in measured
                if item["no_domain_voice_regression"] and item["no_domain_total_regression"]
                and item["mean_voice_eer"] < baseline["mean_voice_eer"] - 1e-12]
    selected = (min(eligible, key=lambda item: (item["mean_voice_eer"], -item["worst_total"]))
                if eligible else baseline)
    payload = {
        "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
        "final_holdout": "NOT RUN",
        "results": results,
        "new_aasist_training": {
            "status": "NOT RUN", "reason": "CUDA unavailable; CPU-only run would not be a controlled practical experiment"},
        "ssl_backbone": {"status": "NOT RUN", "reason": "no licensed offline checkpoint is bundled"},
        "selected": selected["candidate"],
        "decision": ("ADOPT" if eligible else "KEEP_CURRENT_NO_STRICT_ROBUST_IMPROVEMENT"),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.only_current_and_extra and not extras:
        by_name = {item["candidate"]: item for item in results}
        for filename, names in (
            ("voice_augmentation_ablation.json", ("current_spec_cnn", "v7_channel_augmentation")),
            ("partial_fake_ablation.json", ("current_spec_cnn", "v7_partial_fake")),
            ("hard_negative_ablation.json", ("current_spec_cnn", "v7_hard_mining")),
        ):
            (output.parent / filename).write_text(json.dumps({
                "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION", "final_holdout": "NOT RUN",
                "candidates": [by_name[name] for name in names if name in by_name],
                "selected": "current_spec_cnn",
            }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
