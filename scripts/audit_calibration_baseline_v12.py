#!/usr/bin/env python3
"""Explain the historical/fresh V7 calibration baseline discrepancy.

This audit intentionally works only with the old calibration cache, the V11
canonical-specialist cache, and non-final configuration artifacts.  It never
opens the protected final holdout.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import balanced_subset
from src.ensemble import assert_final_holdout_forbidden, score_head_selective_ensemble
from src.models.panns import OFFICIAL_16K_CONFIG


HISTORICAL_CACHE = ROOT / "experiments/v7/fusion_calibration_predictions.csv"
FRESH_SPECIALIST_CACHE = ROOT / "experiments/v11/cache/fusion_calibration.csv"
FULLY_FRESH_CACHE = ROOT / "experiments/v12/cache/v7_canonical/cal_old.csv"
SPLIT = ROOT / "data/splits/fusion_calibration.csv"
WEIGHTS = ROOT / "model/fusion_weights.json"
OUTPUT = ROOT / "experiments/v12/calibration_baseline_audit.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_summary(frame: pd.DataFrame, weights: dict) -> dict:
    folds = {}
    for fold, subset in frame.groupby("calibration_fold", sort=True):
        folds[str(fold)] = score_head_selective_ensemble(subset, weights)
    totals = np.asarray([value["total"] for value in folds.values()], dtype=float)
    return {
        "folds": folds,
        "mean_total": float(totals.mean()),
        "worst_total": float(totals.min()),
        "robust_objective": float(0.7 * totals.mean() + 0.3 * totals.min()),
    }


def frame_identity_sha(frame: pd.DataFrame) -> str:
    columns = [column for column in (
        "path", "file_fake", "voice_fake", "music_fake", "voice_present",
        "music_present", "augment", "data_role", "split_group_id")
        if column in frame]
    payload = frame[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    for path in (HISTORICAL_CACHE, FRESH_SPECIALIST_CACHE, SPLIT, WEIGHTS, OUTPUT):
        assert_final_holdout_forbidden(path)
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    historical_meta = json.loads(HISTORICAL_CACHE.with_suffix(
        HISTORICAL_CACHE.suffix + ".meta.json").read_text(encoding="utf-8"))
    fresh_meta = json.loads(FRESH_SPECIALIST_CACHE.with_suffix(
        FRESH_SPECIALIST_CACHE.suffix + ".meta.json").read_text(encoding="utf-8"))
    historical = pd.read_csv(HISTORICAL_CACHE)
    fresh = pd.read_csv(FRESH_SPECIALIST_CACHE)
    expected_rows = balanced_subset(pd.read_csv(SPLIT), 0, 20260831)

    label_columns = [f"y_{head}" for head in (
        "file_fake", "voice_fake", "music_fake", "voice_present", "music_present")]
    order_columns = ["calibration_fold", "source", "generator", *label_columns]
    order_equal = all(
        historical[column].astype(str).tolist() == fresh[column].astype(str).tolist()
        for column in order_columns)
    value_differences = {}
    for column in ("df_primary", "df_second", "vf", "mf", "vfile", "mfile",
                   "vp_model", "mp_model", "vp_panns", "mp_panns"):
        left = historical[column].to_numpy(dtype=float)
        right = fresh[column].to_numpy(dtype=float)
        value_differences[column] = {
            "mean_absolute_difference": float(np.nanmean(np.abs(left - right))),
            "maximum_absolute_difference": float(np.nanmax(np.abs(left - right))),
            "exactly_equal_with_nan": bool(np.array_equal(left, right, equal_nan=True)),
        }

    artifacts = {
        "voice_checkpoint": ROOT / "model/best.pt",
        "music_checkpoint": ROOT / "model/music_best.pt",
        "panns_checkpoint": ROOT / "model/panns/Cnn14_16k_mAP=0.438.pth",
        "df_arena_checkpoint": ROOT / "model/df_arena/df_arena_1b_int8.onnx",
        "fusion_json": WEIGHTS,
        "split": SPLIT,
        "script": ROOT / "script.py",
    }
    current_sha = {name: sha256(path) for name, path in artifacts.items()}
    report = {
        "status": "MEASURED_NON_FINAL_AUDIT",
        "final_holdout": "NOT RUN",
        "selected_v7_sha256": current_sha,
        "cache_sha256": {
            "historical": sha256(HISTORICAL_CACHE),
            "v11_fresh_specialists": sha256(FRESH_SPECIALIST_CACHE),
        },
        "comparison": {
            "checkpoint_sha": {
                "historical_voice": historical_meta.get("voice_checkpoint_sha256"),
                "fresh_voice": fresh_meta.get("checkpoint_sha256", {}).get("v7_voice"),
                "current_voice": current_sha["voice_checkpoint"],
                "historical_music": historical_meta.get("music_checkpoint_sha256"),
                "fresh_music": fresh_meta.get("checkpoint_sha256", {}).get("v7_music"),
                "current_music": current_sha["music_checkpoint"],
            },
            "panns_sha": {
                "historical_metadata": historical_meta.get("panns_sha256"),
                "current": current_sha["panns_checkpoint"],
            },
            "df_arena_sha": {
                "historical_metadata": historical_meta.get("df_model_sha256"),
                "current": current_sha["df_arena_checkpoint"],
                "note": "The current ONNX serialization SHA differs from the historical metadata; V11 retained historical DF predictions, so this does not cause the 0.888366-to-0.885310 delta but requires fully fresh V12 canonical extraction.",
            },
            "fusion_json_sha": current_sha["fusion_json"],
            "split_sha": {
                "historical_metadata": historical_meta.get("split_csv_sha256", {}).get(
                    "fusion_calibration.csv"),
                "fresh_metadata": fresh_meta.get("split_sha256"),
                "current": current_sha["split"],
            },
            "rows": {
                "historical": len(historical),
                "fresh": len(fresh),
                "split": len(expected_rows),
                "fold_source_generator_label_order_equal": order_equal,
                "historical_path_column": "path" in historical.columns,
                "fresh_path_column": "path" in fresh.columns,
                "expected_selected_row_id_sha256": frame_identity_sha(expected_rows),
                "fresh_selected_row_id_sha256": fresh_meta.get("selected_rows_sha256"),
                "note": "The historical cache omitted path IDs; exact historical path-order verification is impossible. Fold/source/generator/label order is identical and the fresh path digest matches the deterministic split selection.",
            },
            "segment_selection": {
                "historical_metadata": historical_meta.get("feature_extractor_version"),
                "fresh_metadata": fresh_meta.get("aggregation_version"),
                "policy": weights.get("voice_segment_policy"),
                "specialist_max_segments": weights.get("specialist_max_segments"),
            },
            "aggregation": {
                "configured_voice_fake": weights.get("voice_fake_aggregation"),
                "historical_actual_voice_fake": "topk_mean (calibrate_fusion.collect omitted aggregation_config)",
                "fresh_actual_voice_fake": "max",
                "music_fake": "topk_mean",
                "file_fake": "topk_mean",
                "presence": "mean",
            },
            "panns_preprocessing": {
                **OFFICIAL_16K_CONFIG.__dict__,
                "torchlibrosa": True,
                "bn0_active": True,
            },
            "df_crop_policy": {
                "gate": weights.get("df_gate_policy"),
                "adaptive": weights.get("adaptive_df_enabled"),
                "current_submission_primary_only": True,
                "v11_cache_df_source": "historical cached df_primary, not recomputed",
            },
            "sample_rate": 16000,
            "metric_implementation": weights.get("metric_version"),
            "file_fusion": weights.get("file_fusion_mode"),
            "component_fusion": {
                "w_df_voice_component": weights.get("w_df_voice_component"),
                "w_df_music_component": weights.get("w_df_music_component"),
            },
            "cache_version": {
                "historical": historical_meta.get("calibration_script_version"),
                "fresh": fresh_meta.get("schema_version"),
            },
            "pipeline_version": {
                "historical": historical_meta.get("pipeline_version"),
                "fresh": fresh_meta.get("pipeline_version"),
                "selected": weights.get("pipeline_version"),
            },
            "feature_value_differences": value_differences,
        },
        "metrics": {
            "old_historical_cache": robust_summary(historical, weights),
            "v11_fresh_specialist_cache": robust_summary(fresh, weights),
            "v12_fully_fresh_current_artifacts": (
                robust_summary(pd.read_csv(FULLY_FRESH_CACHE), weights)
                if FULLY_FRESH_CACHE.is_file() else "NOT RUN"),
        },
        "WHY_0.888366_BECAME_0.885310": (
            "The historical V7 calibration cache was produced by scripts/calibrate_fusion.py, "
            "whose collect() call did not pass the selected aggregation_config into "
            "infer_wave_features_batch(). The cache therefore used the default top-k-2 mean "
            "for VOICE_FAKE even though its metadata and the selected submission config said "
            "voice_fake_aggregation=max. V11 recomputed the selected V7 voice head on the same "
            "rows/segments with max aggregation, matching the actual submission path. The "
            "resulting ranking changes in vf explain the metric difference. A separate DF ONNX "
            "serialization SHA mismatch was also found; both compared caches use the same "
            "historical DF predictions, so it is not the cause of this delta, but V12 must use "
            "one fully fresh canonical cache for all model components."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "old": report["metrics"]["old_historical_cache"]["robust_objective"],
        "fresh": report["metrics"]["v11_fresh_specialist_cache"]["robust_objective"],
        "voice_vf_mean_abs_diff": value_differences["vf"]["mean_absolute_difference"],
        "df_sha_matches_historical": (
            historical_meta.get("df_model_sha256") == current_sha["df_arena_checkpoint"]),
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
