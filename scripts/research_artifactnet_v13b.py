#!/usr/bin/env python3
"""Leakage-safe ArtifactNet stability and deployment research for V13B.

All choices are made on ``cal_v13b.csv``.  Generator-disjoint data is touched
only once after the CAL policy is frozen.  The script never accesses a final
holdout and never writes a selected submission artifact.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script as submission
from scripts.evaluate_v13b_artifactnet import (
    ARTIFACTNET_SAMPLES, ARTIFACTNET_SR, MODEL, MODEL_DATA,
    baseline_predictions, sha256, to_artifactnet_input,
)
from src.dataset import load_manifest_row_wave
from src.metrics import compute_eer
from src.preprocess import load_audio
from tools.v13_guards import assert_final_holdout_v13b_forbidden


CAL = ROOT / "data/splits_v13b/cal_v13b.csv"
GENERATOR_VAL = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
CEILINGS: tuple[float | None, ...] = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, None)
GATE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
OUTPUT_CAL = ROOT / "experiments/v13b/artifactnet_calibration.json"
OUTPUT_NONFINITE = ROOT / "experiments/v13b/artifactnet_nonfinite_analysis.json"
OUTPUT_GENERATOR = ROOT / "experiments/v13b/artifactnet_generator_confirmation.json"
OUTPUT_GATE = ROOT / "experiments/v13b/artifactnet_selective_gate.json"


def ceiling_name(value: float | None) -> str:
    return "none" if value is None else f"peak_{value:.2f}"


def source_wave(row: pd.Series) -> np.ndarray:
    path = pathlib.Path(str(row["source_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"immutable raw source missing: {path}")
    wave, sample_rate = load_audio(path, target_sr=ARTIFACTNET_SR)
    if sample_rate != ARTIFACTNET_SR:
        raise RuntimeError("ArtifactNet raw-source resample contract broken")
    return wave


def music_rows(path: pathlib.Path) -> pd.DataFrame:
    assert_final_holdout_v13b_forbidden(path)
    frame = pd.read_csv(path)
    frame = frame[frame.music_present.astype(int).eq(1)].copy().reset_index(drop=True)
    if len(frame) == 0 or frame.music_fake.nunique() != 2:
        raise RuntimeError(f"music evaluation requires two classes: {path}")
    return frame


def session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(6, max(1, int(__import__("os").cpu_count() or 1)))
    options.inter_op_num_threads = 1
    value = ort.InferenceSession(str(MODEL), sess_options=options,
                                 providers=["CPUExecutionProvider"])
    contract = [(item.name, item.shape, item.type) for item in value.get_inputs()]
    if contract != [("audio", ["batch", ARTIFACTNET_SAMPLES], "tensor(float)")]:
        raise RuntimeError(f"unexpected ArtifactNet input contract: {contract}")
    return value


def input_stats(before: np.ndarray, after: np.ndarray, *, ceiling: float | None,
                adjusted: bool) -> dict:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    return {
        "ceiling": ceiling, "gain_adjusted": bool(adjusted),
        "input_min": float(before.min()), "input_max": float(before.max()),
        "input_peak": float(np.max(np.abs(before))),
        "input_rms": float(np.sqrt(np.mean(np.square(before)))),
        "input_silence_ratio_abs_le_1e_6": float(np.mean(np.abs(before) <= 1e-6)),
        "post_gain_min": float(after.min()), "post_gain_max": float(after.max()),
        "post_gain_peak": float(np.max(np.abs(after))),
        "post_gain_rms": float(np.sqrt(np.mean(np.square(after)))),
        "sample_rate": ARTIFACTNET_SR, "duration_seconds": 4.0,
    }


def selected_segments(row: pd.Series, maximum: int) -> list[np.ndarray]:
    wave = source_wave(row)
    plan = submission.build_segment_plan(wave, ARTIFACTNET_SR, 4.0)
    duration = len(wave) / float(ARTIFACTNET_SR)
    return submission.select_segments_from_plan(plan, duration, "high_energy", maximum)


def run_scores(frame: pd.DataFrame, *, ceiling: float | None, maximum: int,
               record_all: bool = False) -> tuple[np.ndarray, list[dict], float]:
    """Return per-file scores and segment records without any nonfinite fallback."""
    sess = session()
    started = time.perf_counter()
    scores: list[float] = []
    records: list[dict] = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        values: list[float] = []
        segments = selected_segments(row, maximum)
        for segment_index, segment in enumerate(segments):
            audio, adjusted = to_artifactnet_input(segment, ceiling)
            output = np.asarray(sess.run(None, {"audio": audio.reshape(1, -1)})[0],
                                dtype=np.float64).reshape(-1)
            finite = output.size == 1 and bool(np.isfinite(output).all())
            details = {
                "row_index": row_index, "audio_id": pathlib.Path(str(row["path"])).stem,
                "path": str(row["path"]), "source_path": str(row["source_path"]),
                "generator": str(row.get("generator", "")),
                "music_fake": int(row["music_fake"]), "segment_index": segment_index,
                "output": output.tolist(), "finite": finite,
                **input_stats(segment, audio, ceiling=ceiling, adjusted=adjusted),
            }
            if record_all or not finite:
                records.append(details)
            if not finite:
                values = []
                break
            probability = float(output[0])
            if not 0.0 <= probability <= 1.0:
                details["finite"] = False
                details["reason"] = "probability_outside_unit_interval"
                records.append(details)
                values = []
                break
            values.append(probability)
        scores.append(float("nan") if not values else float(np.median(values)))
    return np.asarray(scores, dtype=np.float64), records, time.perf_counter() - started


def aggregation_scores(frame: pd.DataFrame, *, ceiling: float | None) -> tuple[dict[str, np.ndarray], list[dict], float]:
    """Evaluate only multi-crop aggregations at a fixed CAL frontend.

    The single-crop policy is intentionally evaluated by :func:`run_scores`
    with ``maximum=1``.  Do not run extra segments for it: an invalid *unused*
    segment would otherwise turn a valid one-crop policy into a false failure.
    """
    sess = session(); started = time.perf_counter()
    by_name = {"top2_mean": [], "median": []}
    records: list[dict] = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        values: list[float] = []
        failed = False
        for segment_index, segment in enumerate(selected_segments(row, 3)):
            audio, adjusted = to_artifactnet_input(segment, ceiling)
            output = np.asarray(sess.run(None, {"audio": audio.reshape(1, -1)})[0],
                                dtype=np.float64).reshape(-1)
            finite = output.size == 1 and bool(np.isfinite(output).all())
            if not finite:
                records.append({
                    "row_index": row_index, "audio_id": pathlib.Path(str(row["path"])).stem,
                    "path": str(row["path"]), "source_path": str(row["source_path"]),
                    "generator": str(row.get("generator", "")),
                    "music_fake": int(row["music_fake"]), "segment_index": segment_index,
                    "output": output.tolist(), "finite": False,
                    **input_stats(segment, audio, ceiling=ceiling, adjusted=adjusted),
                })
                failed = True
                break
            values.append(float(output[0]))
        if failed or not values:
            for name in by_name:
                by_name[name].append(float("nan"))
            continue
        array = np.asarray(values, dtype=np.float64)
        by_name["top2_mean"].append(float(np.mean(np.sort(array)[-min(2, len(array)):])) )
        by_name["median"].append(float(np.median(array)))
    return ({key: np.asarray(value) for key, value in by_name.items()}, records,
            time.perf_counter() - started)


def compare_nonfinite_frontends(records: list[dict]) -> list[dict]:
    """Probe only already-observed failing segments under frozen, named frontends.

    This is a root-cause diagnostic, never a deployment choice.  It reports
    whether the bad output follows the individual raw segment or a particular
    peak-conditioning policy without silently dropping that segment.
    """
    if not records:
        return []
    sess = session()
    findings: list[dict] = []
    for record in records:
        # Reconstruct by immutable source and deterministic segment selection;
        # no hidden/test row is involved.
        row = pd.Series({"source_path": record["source_path"], "path": record["path"]})
        segments = selected_segments(row, int(record["segment_index"]) + 1)
        segment = segments[int(record["segment_index"])]
        trials = []
        for candidate_ceiling in CEILINGS:
            audio, adjusted = to_artifactnet_input(segment, candidate_ceiling)
            output = np.asarray(sess.run(None, {"audio": audio.reshape(1, -1)})[0],
                                dtype=np.float64).reshape(-1)
            finite = output.size == 1 and bool(np.isfinite(output).all())
            trials.append({
                "policy": ceiling_name(candidate_ceiling), "ceiling": candidate_ceiling,
                "finite": finite, "output": output.tolist(),
                **input_stats(segment, audio, ceiling=candidate_ceiling, adjusted=adjusted),
            })
        findings.append({
            "audio_id": record["audio_id"], "path": record["path"],
            "source_path": record["source_path"], "segment_index": record["segment_index"],
            "trials": trials,
        })
    return findings


def safe_eer(truth: np.ndarray, score: np.ndarray) -> float | None:
    return None if not np.isfinite(score).all() else compute_eer(truth, score)


def cal_frontend(frame: pd.DataFrame) -> dict:
    truth = frame.music_fake.to_numpy(dtype=int)
    rows = []
    reference = None
    for ceiling in CEILINGS:
        score, records, elapsed = run_scores(frame, ceiling=ceiling, maximum=1)
        if reference is None and np.isfinite(score).all():
            reference = score
        rows.append({
            "ceiling": ceiling, "policy": ceiling_name(ceiling), "maximum_crops": 1,
            "nonfinite_files": int(np.sum(~np.isfinite(score))),
            "nonfinite_segments": len(records), "music_eer": safe_eer(truth, score),
            "runtime_seconds": elapsed,
            "rank_correlation_vs_first_finite": (
                float(spearmanr(reference, score).statistic)
                if reference is not None and np.isfinite(score).all() else None),
        })
    finite = [row for row in rows if row["nonfinite_files"] == 0 and row["music_eer"] is not None]
    if not finite:
        return {"status": "FAIL_NO_FINITE_CAL_FRONTEND", "candidates": rows, "selected": None}
    # Frozen CAL-only objective: lower EER, then less gain intervention, then
    # the declared candidate order.  Generator-disjoint rows are not available here.
    order = {ceiling_name(value): index for index, value in enumerate(CEILINGS)}
    selected = min(finite, key=lambda row: (
        float(row["music_eer"]), order[row["policy"]]))
    return {"status": "PASS_CAL_ONLY", "candidates": rows, "selected": selected}


def presence_scores(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Canonical TEST5 presence prediction, without DF/file fusion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    music = submission.load_music_model(str(device))
    panns = submission.load_panns(str(device))
    config = submission.load_fusion_weights()
    started = time.perf_counter(); predictions = []
    for start in range(0, len(frame), 8):
        chunk = frame.iloc[start:start + 8]
        waves = [load_manifest_row_wave(row, sr=16_000, is_training=False,
                                        use_demucs=False, task="music")
                 for _, row in chunk.iterrows()]
        groups = [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]
        music_output, music_bounds = submission._run_torch_segments(
            music, groups, str(device), use_amp=device.type == "cuda", detector_name="music_presence")
        panns_output, panns_bounds = submission._run_torch_segments(
            panns, groups, str(device), use_amp=False, outputs_are_logits=False,
            detector_name="panns_presence")
        for (ml, mr), (pl, pr) in zip(music_bounds, panns_bounds):
            mp_model = submission.aggregate_head_predictions(
                music_output["music_present"][ml:mr], "music_present", config)
            mp_panns = submission.aggregate_head_predictions(
                panns_output["music_present"][pl:pr], "music_present", config)
            predictions.append(float(config["w_panns_presence"] * mp_panns +
                                     (1.0 - config["w_panns_presence"]) * mp_model))
    return np.asarray(predictions), time.perf_counter() - started


def select_gate(cal_full: pd.DataFrame, cal_music: pd.DataFrame, candidate: np.ndarray,
                baseline: np.ndarray) -> dict:
    presence, elapsed = presence_scores(cal_full)
    mapping = {str(path): score for path, score in zip(cal_full.path, presence)}
    music_presence = np.asarray([mapping[str(path)] for path in cal_music.path])
    truth = cal_music.music_fake.to_numpy(dtype=int)
    rows = []
    for threshold in GATE_THRESHOLDS:
        gated = np.where(music_presence >= threshold, candidate, baseline)
        rows.append({"threshold": threshold, "execution_rate": float(np.mean(presence >= threshold)),
                     "music_present_recall": float(np.mean(music_presence >= threshold)),
                     "music_eer": safe_eer(truth, gated)})
    # CAL-only: minimize EER, then maximize component recall, then minimize calls.
    finite = [row for row in rows if row["music_eer"] is not None]
    if not finite:
        return {"status": "FAIL_NO_FINITE_CAL_GATE", "candidates": rows, "selected": None,
                "presence_runtime_seconds": elapsed}
    selected = min(finite, key=lambda row: (float(row["music_eer"]),
                                            -float(row["music_present_recall"]),
                                            float(row["execution_rate"])))
    return {"status": "PASS_CAL_ONLY", "candidates": rows, "selected": selected,
            "presence_runtime_seconds": elapsed}


def generator_confirmation(policy: dict) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    full = pd.read_csv(GENERATOR_VAL)
    frame = full[full.music_present.astype(int).eq(1)].copy().reset_index(drop=True)
    ceiling = policy["frontend"]["selected"]["ceiling"]
    scores, records, elapsed = run_scores(frame, ceiling=ceiling, maximum=1, record_all=True)
    baseline, baseline_seconds = baseline_predictions(frame, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    payload = {
        "status": "PASS_CONFIRMATION" if np.isfinite(scores).all() else "FAIL_NONFINITE",
        "policy_from_cal_only": policy,
        "rows": len(frame), "music_eer": safe_eer(frame.music_fake.to_numpy(int), scores),
        "baseline_music_eer": compute_eer(frame.music_fake.to_numpy(int), baseline),
        "estimated_music_ads_delta": (None if not np.isfinite(scores).all() else
                                        0.3 * (compute_eer(frame.music_fake.to_numpy(int), baseline) -
                                               compute_eer(frame.music_fake.to_numpy(int), scores))),
        "runtime_seconds": elapsed, "baseline_runtime_seconds": baseline_seconds,
        "segment_records": records,
        "nonfinite_segment_count": int(sum(not bool(record["finite"]) for record in records)),
        "source_disjoint": "NOT MEASURED",
        "final_holdout": "NOT RUN",
    }
    return payload, full, scores, baseline


def main() -> None:
    assert_final_holdout_v13b_forbidden(CAL, GENERATOR_VAL, MODEL, MODEL_DATA,
                                        OUTPUT_CAL, OUTPUT_NONFINITE, OUTPUT_GENERATOR, OUTPUT_GATE)
    if not MODEL.is_file() or not MODEL_DATA.is_file():
        raise FileNotFoundError("ArtifactNet candidate files are required")
    cal_full = pd.read_csv(CAL)
    cal_music = music_rows(CAL)
    frontend = cal_frontend(cal_music)
    report = {
        "status": frontend["status"], "selection_data": "cal_v13b music-present only",
        "model_sha256": sha256(MODEL), "model_data_sha256": sha256(MODEL_DATA),
        "frontend": frontend, "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
    }
    if frontend["selected"] is None:
        OUTPUT_CAL.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("no finite CAL ArtifactNet frontend")
    ceiling = frontend["selected"]["ceiling"]
    one_crop_scores, one_crop_records, one_crop_seconds = run_scores(
        cal_music, ceiling=ceiling, maximum=1, record_all=True)
    if not np.isfinite(one_crop_scores).all():
        raise RuntimeError("CAL-selected ArtifactNet one-crop path is non-finite")
    aggregation, nonfinite_records, aggregation_seconds = aggregation_scores(cal_music, ceiling=ceiling)
    truth = cal_music.music_fake.to_numpy(dtype=int)
    aggregation_rows = [{"name": "highest_energy_one_crop", "maximum_crops": 1,
                         "music_eer": safe_eer(truth, one_crop_scores),
                         "nonfinite_files": int(np.sum(~np.isfinite(one_crop_scores)))}]
    aggregation_rows += [{"name": name, "maximum_crops": 3,
                         "music_eer": safe_eer(truth, values),
                         "nonfinite_files": int(np.sum(~np.isfinite(values)))}
                        for name, values in aggregation.items()]
    report["aggregation"] = {
        "selection_rule": "CAL-only deployment policy fixes highest_energy_one_crop; alternatives reported only",
        "selected": "highest_energy_one_crop", "candidates": aggregation_rows,
        "runtime_seconds": aggregation_seconds,
    }
    report["nonfinite_on_cal"] = nonfinite_records
    # Persist the expensive CAL-only measurement before presence inference;
    # an interrupted later confirmation must not discard the evidence used to
    # reject an unsafe aggregation policy.
    report["selective_gate"] = "PENDING_CAL_PRESENCE_INFERENCE"
    OUTPUT_CAL.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # CAL-only selective gate receives the separately evaluated one-crop score.
    cal_baseline, _ = baseline_predictions(cal_music, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    gate = select_gate(cal_full, cal_music, one_crop_scores, cal_baseline)
    report["one_crop_recheck"] = {
        "status": "PASS", "runtime_seconds": one_crop_seconds,
        "records": one_crop_records,
    }
    report["selective_gate"] = gate
    OUTPUT_CAL.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if gate["selected"] is None:
        raise RuntimeError("no finite CAL ArtifactNet selective-gate policy")

    # Generator-disjoint is read after every deploy policy has been frozen above.
    frozen = {"frontend": frontend, "aggregation": report["aggregation"],
              "selective_gate": gate}
    confirmation, generator_full, generator_scores, generator_baseline = generator_confirmation(frozen)
    OUTPUT_GENERATOR.write_text(json.dumps(confirmation, indent=2) + "\n", encoding="utf-8")

    # Nonfinite root-cause diagnostic is deliberately separate from the one-crop
    # confirmation. It does not contribute to any choice.
    diagnostic_frame = music_rows(GENERATOR_VAL)
    diagnostic_scores, diagnostic_records, diagnostic_seconds = run_scores(
        diagnostic_frame, ceiling=ceiling, maximum=3, record_all=False)
    nonfinite = [record for record in diagnostic_records if not record["finite"]]
    frontend_comparison = compare_nonfinite_frontends(nonfinite)
    OUTPUT_NONFINITE.write_text(json.dumps({
        "status": "MEASURED", "policy_frozen_from_cal": ceiling_name(ceiling),
        "nonfinite_count": len(nonfinite), "records": nonfinite,
        "same_segment_all_frontend_trials": frontend_comparison,
        "diagnostic_runtime_seconds": diagnostic_seconds,
        "model_intermediates": "NOT AVAILABLE: released ONNX exposes only final prob output",
        "root_cause": {
            "measured": "nonfinite output occurs after finite, non-silent 44.1 kHz input reaches the opaque ONNX graph; same-segment frontend trials are recorded above",
            "not_established": ["specific internal operator", "weight corruption", "normalization denominator"],
            "production_consequence": "FAIL_CLOSED; direct deployment is rejected if any segment is non-finite",
        }, "final_holdout": "NOT RUN",
    }, indent=2) + "\n", encoding="utf-8")

    # Apply the CAL-selected gate once to generator-disjoint confirmation.
    threshold = gate["selected"]["threshold"]
    presence, presence_seconds = presence_scores(generator_full)
    presence_by_path = {str(path): value for path, value in zip(generator_full.path, presence)}
    music_presence = np.asarray([presence_by_path[str(path)] for path in diagnostic_frame.path])
    gated = np.where(music_presence >= threshold, generator_scores, generator_baseline)
    OUTPUT_GATE.write_text(json.dumps({
        "status": "GENERATOR_CONFIRMATION_ONLY", "threshold_frozen_from_cal": threshold,
        "execution_rate_generator_full": float(np.mean(presence >= threshold)),
        "music_present_recall_generator": float(np.mean(music_presence >= threshold)),
        "music_eer_generator": safe_eer(diagnostic_frame.music_fake.to_numpy(int), gated),
        "candidate_added_runtime_minutes_projected": (
            float(confirmation["runtime_seconds"]) / len(diagnostic_frame) *
            float(np.mean(presence >= threshold)) * 20.0),
        "presence_runtime_seconds": presence_seconds,
        "cps_delta": "NOT MEASURED: presence scores are unchanged",
        "file_delta": "NOT MEASURED: requires independent fusion calibration",
        "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cal_frontend": frontend["selected"], "cal_gate": gate["selected"],
                      "generator": confirmation["status"]}, indent=2))


if __name__ == "__main__":
    main()
