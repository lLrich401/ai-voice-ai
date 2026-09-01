#!/usr/bin/env python3
"""Compare a promising new FILE representation with canonical TEST5 FILE scores."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_eer
from src.models.music_forensic import PANNsForensicHead
from scripts.train_v13b_music_representations import predict_panns_head
from tools.v13_guards import assert_final_holdout_v13b_forbidden

import script as submission


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eer_predictions(truth: np.ndarray, score: np.ndarray) -> np.ndarray:
    fpr, tpr, thresholds = roc_curve(truth, score, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return (score >= thresholds[index]).astype(int)


def complementarity(truth: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    a = eer_predictions(truth, first) == truth
    b = eer_predictions(truth, second) == truth
    return {
        "both_correct": int(np.sum(a & b)),
        "both_wrong": int(np.sum(~a & ~b)),
        "only_canonical_wrong": int(np.sum(~a & b)),
        "only_candidate_wrong": int(np.sum(a & ~b)),
        "classification_note": "each detector thresholded at its own closest empirical EER point",
    }


def main() -> None:
    split_path = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
    candidate_path = ROOT / "model/candidates/v13b/m2_music_panns_frozen_probe.pt"
    output_path = ROOT / "experiments/v13b/m2_file_complementarity.json"
    assert_final_holdout_v13b_forbidden(split_path, candidate_path, output_path)
    frame = pd.read_csv(split_path)
    paths = [pathlib.Path(value) for value in frame.path]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("FILE complementarity requires direct local validation audio")
    if frame.path.map(lambda value: pathlib.Path(value).stem).duplicated().any():
        raise RuntimeError("duplicate validation stems prevent exact mapping")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    submission.verify_mandatory_models()
    voice = submission.load_voice_model(device)
    music = submission.load_music_model(device)
    df_model = submission.load_df_arena(device)
    panns = submission.load_panns(device)
    weights = submission.load_fusion_weights()
    started = time.perf_counter(); rows = []
    for start in range(0, len(paths), 4):
        rows.extend(submission.infer_files_batch(
            voice, music, df_model, panns, weights, paths[start:start + 4], device,
            use_demucs=False))
    canonical_seconds = time.perf_counter() - started
    by_id = {str(row[0]): float(row[1]) for row in rows}
    canonical = np.asarray([by_id[path.stem] for path in paths])
    checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if checkpoint.get("candidate") != "M2" or not checkpoint.get("backbone_frozen"):
        raise RuntimeError("M2 FILE checkpoint metadata mismatch")
    head = PANNsForensicHead().to(device)
    head.load_state_dict(checkpoint["model"], strict=True); head.eval()
    candidate_started = time.perf_counter()
    candidate = predict_panns_head(head, panns, frame, torch.device(device))["file_fake"]
    candidate_seconds = time.perf_counter() - candidate_started
    truth = frame.file_fake.to_numpy(int)
    canonical_eer = compute_eer(truth, canonical)
    candidate_eer = compute_eer(truth, candidate)
    fusion_grid = []
    for candidate_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        score = (1.0 - candidate_weight) * canonical + candidate_weight * candidate
        fusion_grid.append({
            "candidate_weight": candidate_weight,
            "file_eer": compute_eer(truth, score),
        })
    best = min(fusion_grid, key=lambda row: row["file_eer"])
    payload = {
        "status": "MEASURED_GENERATOR_DISJOINT_EXPLORATORY_NOT_ADOPTABLE",
        "rows": len(frame), "device": device,
        "canonical_test5_file_eer": canonical_eer,
        "m2_candidate_file_eer": candidate_eer,
        "fusion_grid": fusion_grid,
        "best_same_split_exploratory_fusion": best,
        "same_split_selection_warning": "not independent calibration; cannot be adopted",
        "correlation": {
            "pearson": float(pearsonr(canonical, candidate).statistic),
            "spearman": float(spearmanr(canonical, candidate).statistic),
        },
        "error_overlap": complementarity(truth, canonical, candidate),
        "ads_contributions": {
            "candidate_file_vs_canonical": 0.5 * (canonical_eer - candidate_eer),
            "best_fusion_file_vs_canonical": 0.5 * (canonical_eer - best["file_eer"]),
            "music": 0.0, "voice": 0.0,
        },
        "runtime": {
            "canonical_total_seconds": canonical_seconds,
            "canonical_seconds_per_file": canonical_seconds / len(frame),
            "candidate_total_seconds": candidate_seconds,
            "candidate_seconds_per_file": candidate_seconds / len(frame),
            "candidate_added_1200_file_minutes_projected": candidate_seconds / len(frame) * 20,
        },
        "dependencies": {
            "split_sha256": sha256(split_path), "candidate_sha256": sha256(candidate_path),
            "fusion_sha256": sha256(ROOT / "model/fusion_weights.json"),
        },
        "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
        "decision": "KEEP_TEST5",
    }
    prediction_path = ROOT / "experiments/v13b/m2_file_complementarity_scores.csv"
    pd.DataFrame({
        "audio_id": [path.stem for path in paths], "file_fake": truth,
        "canonical_test5_file_probability": canonical,
        "m2_file_probability": candidate,
    }).to_csv(prediction_path, index=False)
    payload["scores_file"] = prediction_path.relative_to(ROOT).as_posix()
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
