#!/usr/bin/env python3
"""Recalibrate fusion for an eligible music candidate without mutating submission files."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script as submission
from scripts.calibrate_fusion import balanced_subset, robust_score
from src.dataset import load_manifest_row_wave
from src.models.beats_backbone import MusicMultitask


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(path: pathlib.Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("task") != "music" or checkpoint.get("backbone") != "spec_cnn":
        raise RuntimeError("candidate is not a strict music SpecCNN checkpoint")
    model = MusicMultitask(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def replace_music_features(cache, frame, model, device, batch_size):
    updated = cache.copy()
    outputs = {"mf": [], "mfile": [], "mp_model": []}
    for start in range(0, len(frame), batch_size):
        rows = frame.iloc[start:start + batch_size]
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False,
                                        use_demucs=False, task="music")
                 for _, row in rows.iterrows()]
        groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
        result, bounds = submission._run_torch_segments(model, groups, device, use_amp=True)
        for a, b in bounds:
            outputs["mf"].append(submission.aggregate_predictions(
                result["music_fake"][a:b], "topk_mean", 2))
            outputs["mfile"].append(submission.aggregate_predictions(
                result["file_fake"][a:b], "topk_mean", 2))
            outputs["mp_model"].append(float(np.mean(result["music_present"][a:b])))
    for key, values in outputs.items():
        updated[key] = values
    return updated


def coordinate_search(cache, initial):
    detector = [(a / 4, b / 4, (4 - a - b) / 4)
                for a in range(5) for b in range(5 - a)]
    groups = (
        ("detector", detector),
        ("w_df_arena", (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("w_df_voice_component", (0.0, 0.05, 0.10, 0.20, 0.30)),
        ("w_df_music_component", (0.0, 0.025, 0.05, 0.10, 0.20)),
        ("w_panns_presence", (0.0, 0.25, 0.5, 0.75, 1.0)),
    )
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    selected = dict(initial)
    trace = []
    for pass_index in range(2):
        for name, values in groups:
            best = None
            for value in values:
                trial = dict(selected)
                if name == "detector":
                    trial.update(dict(zip(
                        ("w_voice_file", "w_music_file", "w_prob_or"), value)))
                    complexity = 0.0
                else:
                    trial[name] = value
                    complexity = float(value) if "component" in name else 0.0
                objective, metrics = robust_score(cache, trial, adaptive)
                candidate = (objective, -complexity, trial, metrics)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            objective, _, selected, metrics = best
            trace.append({"pass": pass_index + 1, "coordinate": name,
                          "objective": objective, "weights": dict(selected)})
    return selected, objective, metrics, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="model/candidates/music_spec_cnn_v9.pt")
    parser.add_argument("--cache", default="experiments/v7/fusion_calibration_predictions.csv")
    parser.add_argument("--output", default="experiments/v9/fusion_candidate_calibration.json")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    frame = balanced_subset(pd.read_csv("data/splits/fusion_calibration.csv"), 0, 20260831)
    cache = pd.read_csv(args.cache)
    if len(frame) != len(cache):
        raise RuntimeError("calibration cache/frame length mismatch")
    for source, cached in (("file_fake", "y_file_fake"), ("voice_fake", "y_voice_fake"),
                           ("music_fake", "y_music_fake"),
                           ("voice_present", "y_voice_present"),
                           ("music_present", "y_music_present")):
        if not np.array_equal(frame[source].to_numpy(dtype=float),
                              cache[cached].to_numpy(dtype=float)):
            raise RuntimeError(f"calibration cache order mismatch: {source}")
    candidate_path = pathlib.Path(args.candidate)
    model = load_candidate(candidate_path, device)
    candidate_cache = replace_music_features(cache, frame, model, device, args.batch_size)
    configured = json.loads(pathlib.Path("model/fusion_weights.json").read_text(encoding="utf-8"))
    initial = {key: configured[key] for key in (
        "w_voice_file", "w_music_file", "w_prob_or", "w_df_arena",
        "w_df_voice_component", "w_df_music_component", "w_panns_presence",
        "file_fusion_mode",
    )}
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    before_current, current_folds = robust_score(cache, initial, adaptive)
    before_candidate, candidate_initial_folds = robust_score(candidate_cache, initial, adaptive)
    weights, objective, folds, trace = coordinate_search(candidate_cache, initial)
    weights.update({"df_gate_policy": "off", "adaptive_df_enabled": False,
                    "music_checkpoint_sha256": sha256(candidate_path)})
    payload = {
        "status": "MEASURED_INDEPENDENT_CALIBRATION_CANDIDATE_ONLY",
        "final_holdout": "NOT RUN", "selected_submission_files_mutated": False,
        "current_model_current_weights_objective": before_current,
        "candidate_current_weights_objective": before_candidate,
        "candidate_recalibrated_objective": objective,
        "current_folds": current_folds, "candidate_initial_folds": candidate_initial_folds,
        "candidate_recalibrated_folds": folds, "candidate_weights": weights,
        "coordinate_trace": trace,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
