#!/usr/bin/env python3
"""CAL-only F2 FILE blend screen; never edits the frozen TEST5 fusion.

The prior F2 blend observation used generator-disjoint rows and is intentionally
not a selection.  This script makes one small, predeclared blend decision from
the independent V13B calibration split and records it as *not adoptable* until
source-disjoint validation, bootstrap, licence, and runtime gates pass.
"""

from __future__ import annotations

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

from scripts.train_v13b_music_representations import predict_panns_head
from src.metrics import compute_eer
from src.models.music_forensic import PANNsForensicHead
from tools.v13_guards import assert_final_holdout_v13b_forbidden

import script as submission


CAL = ROOT / "data/splits_v13b/cal_v13b.csv"
CHECKPOINT = ROOT / "model/candidates/v13b/m2_music_panns_frozen_probe.pt"
OUTPUT = ROOT / "experiments/v13b/f2_calibration.json"
SCORES = ROOT / "experiments/v13b/f2_calibration_scores.csv"
WEIGHTS = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_file_scores(frame: pd.DataFrame, device: str) -> tuple[np.ndarray, float]:
    paths = [pathlib.Path(value) for value in frame.path]
    if not all(path.is_file() for path in paths):
        raise RuntimeError("CAL needs directly readable local audio")
    if frame.path.map(lambda value: pathlib.Path(value).stem).duplicated().any():
        raise RuntimeError("duplicate CAL stems prevent exact ID mapping")
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
    by_id = {str(row[0]): float(row[1]) for row in rows}
    if set(by_id) != {path.stem for path in paths}:
        raise RuntimeError("canonical CAL prediction ID mismatch")
    return np.asarray([by_id[path.stem] for path in paths]), time.perf_counter() - started


def main() -> None:
    assert_final_holdout_v13b_forbidden(CAL, CHECKPOINT, OUTPUT, SCORES)
    frame = pd.read_csv(CAL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    canonical, canonical_seconds = canonical_file_scores(frame, device)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint.get("candidate") != "M2" or not checkpoint.get("backbone_frozen"):
        raise RuntimeError("expected frozen M2 F2 checkpoint")
    panns = submission.load_panns(device)
    head = PANNsForensicHead().to(device)
    head.load_state_dict(checkpoint["model"], strict=True); head.eval()
    started = time.perf_counter()
    f2 = predict_panns_head(head, panns, frame, torch.device(device))["file_fake"]
    f2_seconds = time.perf_counter() - started
    if not (np.isfinite(canonical).all() and np.isfinite(f2).all()):
        raise RuntimeError("nonfinite F2 calibration predictions")
    truth = frame.file_fake.to_numpy(int)
    candidates = []
    for weight in WEIGHTS:
        blended = (1.0 - weight) * canonical + weight * f2
        candidates.append({"f2_weight": weight, "file_eer": compute_eer(truth, blended)})
    baseline = candidates[0]
    # Fixed CAL-only objective: improve strictly over the canonical output;
    # ties prefer lower complexity (smaller added F2 weight).
    selected = min(candidates, key=lambda row: (row["file_eer"], row["f2_weight"]))
    selected_status = ("CAL_SIGNAL_ONLY" if selected["file_eer"] < baseline["file_eer"]
                       else "REJECT_NO_STRICT_CAL_IMPROVEMENT")
    pd.DataFrame({
        "audio_id": [pathlib.Path(value).stem for value in frame.path],
        "file_fake": truth, "canonical_test5_file_probability": canonical,
        "f2_file_probability": f2,
    }).to_csv(SCORES, index=False)
    payload = {
        "status": selected_status,
        "selection_data": "cal_v13b only",
        "rows": len(frame), "device": device,
        "candidates": candidates, "selected": selected,
        "baseline": baseline,
        "strict_cal_file_eer_delta": baseline["file_eer"] - selected["file_eer"],
        "selection_constraints": [
            "no generator-disjoint score was used to choose the weight",
            "no final holdout was read",
            "frozen TEST5 fusion_weights.json was not edited",
            "source-disjoint, bootstrap, runtime, and adoption data gates remain required",
        ],
        "runtime": {
            "canonical_seconds": canonical_seconds, "f2_seconds": f2_seconds,
            "f2_added_1200_file_minutes_projected": f2_seconds / len(frame) * 20.0,
        },
        "dependencies": {
            "cal_sha256": sha256(CAL), "f2_checkpoint_sha256": sha256(CHECKPOINT),
            "frozen_fusion_sha256": sha256(ROOT / "model/fusion_weights.json"),
        },
        "scores_file": SCORES.relative_to(ROOT).as_posix(),
        "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
        "selected_artifacts_mutated": False, "decision": "KEEP_TEST5",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
