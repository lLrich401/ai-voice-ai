#!/usr/bin/env python3
"""Submission-equivalent non-final evaluation for V13B exploratory SpecCNNs."""

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
from src.metrics import compute_dacon_metrics
from src.models.beats_backbone import MusicMultitask
from tools.v13_guards import assert_final_holdout_v13b_forbidden


HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: pathlib.Path, device: torch.device) -> tuple[MusicMultitask, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("backbone") != "spec_cnn":
        raise RuntimeError(f"V13B exploratory evaluator requires SpecCNN: {path}")
    if tuple(checkpoint.get("label_heads", ())) != HEADS:
        raise RuntimeError(f"checkpoint label heads mismatch: {path}")
    model = MusicMultitask(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def evaluate_checkpoint(path: pathlib.Path, frame: pd.DataFrame,
                        device: torch.device, name: str) -> dict:
    assert_final_holdout_v13b_forbidden(path, "data/splits_v13b/val_generator_disjoint.csv")
    model, checkpoint = load_model(path, device)
    started = time.perf_counter()
    waves = [load_manifest_row_wave(row, sr=16_000, is_training=False,
                                    use_demucs=False, task="music")
             for _, row in frame.iterrows()]
    groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
    outputs, bounds = submission._run_torch_segments(
        model, groups, device, use_amp=device.type == "cuda", detector_name=name)
    predicted = {head: [] for head in HEADS}
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    for left, right in bounds:
        for head in HEADS:
            predicted[head].append(submission.aggregate_head_predictions(
                outputs[head][left:right], head, weights))
    truth = {head: frame[head].to_numpy(int) for head in HEADS}
    metrics = compute_dacon_metrics(truth, {key: np.asarray(value) for key, value in predicted.items()})
    return {
        "name": name, "checkpoint": path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(path), "checkpoint_epoch": checkpoint.get("epoch"),
        "status": "MEASURED_GENERATOR_DISJOINT_ONLY_NOT_ADOPTABLE",
        "metrics": metrics, "runtime_seconds": time.perf_counter() - started,
        "rows": len(frame), "segment_policy": "submission high_energy adaptive 1/2/3",
        "aggregation": "model/fusion_weights.json canonical head aggregation",
    }


def compare(baseline: dict, candidate: dict, task: str) -> dict:
    base = baseline["metrics"]
    new = candidate["metrics"]
    result = {
        "task": task,
        "file_eer_baseline": base["file_eer"], "file_eer_candidate": new["file_eer"],
        "music_eer_baseline": base["music_eer"], "music_eer_candidate": new["music_eer"],
        "delta_music_eer": new["music_eer"] - base["music_eer"],
        "estimated_ads_contribution_music": 0.3 * (base["music_eer"] - new["music_eer"]),
        "standalone_file_delta": new["file_eer"] - base["file_eer"],
        "standalone_file_warning": "not a direct replacement for selected DF-Arena/fusion FILE output",
    }
    target = "music_eer" if task == "music" else "file_eer"
    result["exploratory_improved"] = new[target] < base[target]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("music", "file"), required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    validation_path = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
    assert_final_holdout_v13b_forbidden(validation_path, args.output)
    validation = pd.read_csv(validation_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_path = ROOT / "model/music_best.pt"
    candidate_path = (args.candidate if args.candidate.is_absolute()
                      else ROOT / args.candidate).resolve()
    results = [
        evaluate_checkpoint(baseline_path, validation, device, "TEST5_MUSIC_SPECCNN"),
        evaluate_checkpoint(candidate_path, validation, device, f"V13B_{args.task.upper()}_CANDIDATE"),
    ]
    payload = {
        "status": "EXPLORATORY_MEASURED_NOT_ADOPTABLE",
        "device": str(device), "validation": "generator-disjoint only",
        "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
        "results": results, "comparison": compare(results[0], results[1], args.task),
        "decision": "KEEP_TEST5_REGARDLESS_OF_EXPLORATORY_RESULT",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
