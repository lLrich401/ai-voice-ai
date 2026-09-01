#!/usr/bin/env python3
"""Evaluate the pinned ArtifactNet music-forensics candidate without final-holdout access.

The released ONNX graph accepts a nominally dynamic input but its final linear
layer is batch-one only.  This evaluator therefore runs one 44.1 kHz, four
second chunk at a time and rejects non-finite outputs instead of silently
turning them into valid-looking probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script as submission
from src.dataset import load_manifest_row_wave
from src.metrics import compute_eer
from src.models.beats_backbone import MusicMultitask
from src.preprocess import load_audio
from tools.v13_guards import assert_final_holdout_v13b_forbidden


MODEL = ROOT / "model/candidates/v13b/artifactnet/artifactnet_v94_full.onnx"
MODEL_DATA = ROOT / "model/candidates/v13b/artifactnet/artifactnet_v94_full.onnx.data"
SPLIT = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
OUTPUT = ROOT / "experiments/v13b/artifactnet_generator_disjoint.json"
SCORES = ROOT / "experiments/v13b/artifactnet_generator_disjoint_scores.csv"
HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
ARTIFACTNET_SR = 44_100
ARTIFACTNET_SAMPLES = 4 * ARTIFACTNET_SR


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_artifactnet_input(segment: np.ndarray) -> tuple[np.ndarray, bool]:
    """Make a four-second input and apply a fixed numerical-stability ceiling.

    The released graph produces NaN for some ordinary full-scale material.  A
    fixed linear gain ceiling at 0.25 avoids that singularity without clipping,
    changing ranks, or consulting labels.  The adjustment is reported because
    it is not part of the short upstream usage example.
    """
    wave = np.asarray(segment, dtype=np.float32).reshape(-1)
    if not np.isfinite(wave).all():
        raise RuntimeError("ArtifactNet input contains non-finite samples")
    if len(wave) < ARTIFACTNET_SAMPLES:
        converted = np.pad(wave, (0, ARTIFACTNET_SAMPLES - len(wave)))
    else:
        converted = wave[:ARTIFACTNET_SAMPLES]
    peak = float(np.max(np.abs(converted)))
    adjusted = peak > 0.25
    if adjusted:
        converted = converted * np.float32(0.25 / peak)
    if peak == 0.0:
        converted = converted.copy()
        converted[0] = np.finfo(np.float32).eps
    return np.ascontiguousarray(converted, dtype=np.float32), adjusted


def artifactnet_predictions(frame: pd.DataFrame, *, diagnostic_skip_nonfinite: bool = False) -> tuple[
        dict[str, np.ndarray], float, int, int, int, int]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(6, max(1, int(__import__("os").cpu_count() or 1)))
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])
    if [(value.name, value.shape) for value in session.get_inputs()] != [
            ("audio", ["batch", ARTIFACTNET_SAMPLES])]:
        raise RuntimeError("unexpected ArtifactNet input contract")
    started = time.perf_counter()
    segment_count = 0
    gain_adjusted = 0
    invalid_segments = 0
    all_invalid_files = 0
    by_aggregation = {key: [] for key in (
        "median", "mean", "max", "top2_mean", "highest_energy_one_crop")}
    for _, row in frame.iterrows():
        # V13B stores a canonical 16 kHz derivative in `path`, but this
        # forensics model specifically needs the row's immutable raw source.
        # Production inference would decode the evaluator input path directly.
        source_path = pathlib.Path(str(row["source_path"]))
        if not source_path.is_file():
            raise FileNotFoundError(f"missing immutable source audio: {source_path}")
        wave, _ = load_audio(source_path, target_sr=ARTIFACTNET_SR)
        segments = submission.select_aux_segments(
            wave, sr=ARTIFACTNET_SR, seg_sec=4.0, policy="high_energy")
        probabilities = []
        probability_energies = []
        for segment in segments:
            audio, adjusted = to_artifactnet_input(segment)
            gain_adjusted += int(adjusted)
            output = np.asarray(
                session.run(None, {"audio": audio.reshape(1, -1)})[0],
                dtype=np.float64).reshape(-1)
            if output.size != 1 or not np.isfinite(output).all():
                if not diagnostic_skip_nonfinite:
                    raise RuntimeError(
                        f"ArtifactNet produced non-finite output for {row['path']}: {output}")
                invalid_segments += 1
                continue
            probability = float(output[0])
            if not 0.0 <= probability <= 1.0:
                raise RuntimeError(f"ArtifactNet probability outside [0,1]: {probability}")
            probabilities.append(probability)
            probability_energies.append(float(np.mean(np.square(segment, dtype=np.float64))))
            segment_count += 1
        if not probabilities:
            # Fail neutral rather than manufacturing a positive or negative
            # rank.  The event is counted and prevents automatic adoption.
            probabilities.append(0.5)
            probability_energies.append(0.0)
            all_invalid_files += 1
        values = np.asarray(probabilities, dtype=np.float64)
        by_aggregation["median"].append(float(np.median(values)))
        by_aggregation["mean"].append(float(np.mean(values)))
        by_aggregation["max"].append(float(np.max(values)))
        by_aggregation["top2_mean"].append(float(np.mean(np.sort(values)[-min(2, len(values)):])))
        by_aggregation["highest_energy_one_crop"].append(
            float(values[int(np.argmax(probability_energies))]))
    return ({key: np.asarray(value) for key, value in by_aggregation.items()},
            time.perf_counter() - started, segment_count, gain_adjusted,
            invalid_segments, all_invalid_files)


def load_baseline(device: torch.device) -> MusicMultitask:
    checkpoint = torch.load(ROOT / "model/music_best.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("backbone") != "spec_cnn" or tuple(checkpoint.get("label_heads", ())) != HEADS:
        raise RuntimeError("selected music checkpoint contract mismatch")
    model = MusicMultitask(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def baseline_predictions(frame: pd.DataFrame, device: torch.device) -> tuple[np.ndarray, float]:
    model = load_baseline(device)
    started = time.perf_counter()
    waves = [load_manifest_row_wave(
        row, sr=16_000, is_training=False, use_demucs=False, task="music")
             for _, row in frame.iterrows()]
    groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
    outputs, bounds = submission._run_torch_segments(
        model, groups, device, use_amp=device.type == "cuda", detector_name="TEST5_MUSIC_SPECCNN")
    config = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    scores = [submission.aggregate_head_predictions(
        outputs["music_fake"][left:right], "music_fake", config) for left, right in bounds]
    return np.asarray(scores, dtype=np.float64), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-skip-nonfinite", action="store_true",
        help="diagnostic only: skip invalid segments; never use for production selection")
    args = parser.parse_args()
    assert_final_holdout_v13b_forbidden(SPLIT, MODEL, MODEL_DATA, OUTPUT, SCORES)
    if not MODEL.is_file() or not MODEL_DATA.is_file():
        raise FileNotFoundError("pinned ArtifactNet ONNX and external data file are required")
    full = pd.read_csv(SPLIT)
    frame = full[full.music_present.astype(int).eq(1)].copy().reset_index(drop=True)
    if len(frame) == 0 or frame.music_fake.nunique() != 2:
        raise RuntimeError("music-present generator-disjoint evaluation needs both classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, baseline_seconds = baseline_predictions(frame, device)
    (candidate, candidate_seconds, segment_count, gain_adjusted,
     invalid_segments, all_invalid_files) = artifactnet_predictions(
         frame, diagnostic_skip_nonfinite=args.diagnostic_skip_nonfinite)
    truth = frame.music_fake.to_numpy(dtype=int)
    baseline_eer = compute_eer(truth, baseline)
    candidate_eers = {key: compute_eer(truth, value) for key, value in candidate.items()}
    official = candidate["median"]
    blend_grid = []
    for candidate_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = (1.0 - candidate_weight) * baseline + candidate_weight * official
        blend_grid.append({
            "artifactnet_weight": candidate_weight,
            "music_eer": compute_eer(truth, blended),
        })
    rows = pd.DataFrame({
        "audio_id": [pathlib.Path(value).stem for value in frame.path],
        "dataset": frame.dataset.astype(str),
        "generator": frame.generator.astype(str),
        "music_fake": truth,
        "test5_music_probability": baseline,
        **{f"artifactnet_{key}": value for key, value in candidate.items()},
    })
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SCORES, index=False)
    official_eer = candidate_eers["median"]
    strong = official_eer <= 0.28
    payload = {
        "status": ("DIAGNOSTIC_GENERATOR_DISJOINT_NONFINITE_SKIPPED" if invalid_segments
                   else "MEASURED_GENERATOR_DISJOINT_CANDIDATE"),
        "validation": "data/splits_v13b/val_generator_disjoint.csv; music_present==1",
        "rows": len(frame),
        "class_counts": {str(key): int(value) for key, value in frame.music_fake.value_counts().sort_index().items()},
        "generators": {str(key): int(value) for key, value in frame.generator.value_counts().items()},
        "model": {
            "name": "ArtifactNet v9.4 full ONNX", "upstream_revision": "7c9b753a9d006b48e4bfaf85bf0157e135f4aad4",
            "onnx_sha256": sha256(MODEL), "external_data_sha256": sha256(MODEL_DATA),
            "license": "CC-BY-NC-4.0; research/non-commercial only; upstream patent notice applies",
            "input": "44.1 kHz mono, 4 seconds, batch=1", "output": "P(AI)",
            "candidate_frontend": "immutable raw source; linear peak ceiling 0.25 for ONNX numerical stability",
        },
        "baseline_music_eer": baseline_eer,
        "artifactnet_music_eer": candidate_eers,
        "primary_predeclared_aggregation": "median (upstream song-level recommendation)",
        "fast_policy": {
            "name": "highest_energy_one_crop",
            "music_eer": candidate_eers["highest_energy_one_crop"],
            "projected_added_minutes_if_run_on_all_1200": (
                candidate_seconds / max(1, segment_count + invalid_segments) * 20),
            "note": "same evaluation split diagnostic; requires independent confirmation",
        },
        "estimated_music_ads_delta_vs_test5": 0.3 * (baseline_eer - official_eer),
        "same_split_exploratory_blend": blend_grid,
        "same_split_selection_warning": "blend weights and alternative aggregations are diagnostic only, not adoptable calibration",
        "nonfinite_policy": ("DIAGNOSTIC_SKIP_EXPLICITLY_ENABLED" if args.diagnostic_skip_nonfinite
                             else "FAIL_CLOSED"),
        "correlation": {
            "pearson": float(pearsonr(baseline, official).statistic),
            "spearman": float(spearmanr(baseline, official).statistic),
        },
        "runtime": {
            "device_baseline": str(device), "segments": segment_count,
            "gain_adjusted_segments": gain_adjusted,
            "invalid_segments_skipped": invalid_segments,
            "all_invalid_files_neutral_fallback": all_invalid_files,
            "test5_seconds": baseline_seconds, "artifactnet_seconds": candidate_seconds,
            "artifactnet_seconds_per_music_file": candidate_seconds / len(frame),
            "projected_added_minutes_if_run_on_all_1200": candidate_seconds / len(frame) * 20,
        },
        "gates": {
            "strong_generator_disjoint_threshold": 0.28,
            "excellent_generator_disjoint_threshold": 0.25,
            "strong_pass": strong,
            "source_disjoint": "NOT MEASURED",
            "license_competition_approval": "NOT CONFIRMED",
            "numerical_stability": "PASS" if invalid_segments == 0 else "FAIL",
        },
        "decision": ("REJECT_PRODUCTION_NONFINITE" if invalid_segments else
                     ("CANDIDATE_ONLY_PENDING_SOURCE_DISJOINT_AND_LICENSE" if strong
                      else "REJECT_KEEP_TEST5")),
        "final_holdout": "NOT RUN",
        "scores_file": SCORES.relative_to(ROOT).as_posix(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
