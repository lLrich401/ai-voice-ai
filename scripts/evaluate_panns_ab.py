#!/usr/bin/env python3
"""Evaluate the legacy and upstream-compatible PANNs presence paths.

This script uses only labelled local validation splits. It never reads DACON
test audio and does not tune fake-detector scores.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
from src.metrics import compute_auc
from src.models.panns import OFFICIAL_16K_CONFIG, PANNsPresenceWrapper
import script as submission


def balanced_order(frame: pd.DataFrame, seed: int = 20260831) -> pd.DataFrame:
    """Match full-split ordering used by calibrate_fusion.py."""
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def metrics(frame: pd.DataFrame, voice_column: str, music_column: str) -> dict[str, float]:
    voice_auc = compute_auc(frame["voice_present"], frame[voice_column])
    music_auc = compute_auc(frame["music_present"], frame[music_column])
    return {
        "voice_auc": voice_auc,
        "music_auc": music_auc,
        "cps": 0.5 * (voice_auc + music_auc),
    }


def infer_split(model, frame: pd.DataFrame, device: torch.device, batch_size: int,
                max_segments: int) -> tuple[pd.DataFrame, float]:
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for start in range(0, len(frame), batch_size):
        rows = frame.iloc[start:start + batch_size]
        waves = [load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)
                 for _, row in rows.iterrows()]
        groups = [submission.limit_aux_segments(
            submission.select_aux_segments(wave, sr=16000, seg_sec=4.0), max_segments)
            for wave in waves]
        outputs, bounds = submission._run_torch_segments(
            model, groups, device, use_amp=False, outputs_are_logits=False)
        for row_index, (_, row) in enumerate(rows.iterrows()):
            lower, upper = bounds[row_index]
            records.append({
                "path": str(row["path"]),
                "source": str(row.get("source", "unknown")),
                "generator": str(row.get("generator", "unknown")),
                "calibration_fold": str(row.get("calibration_fold", "")),
                "voice_present": int(row["voice_present"]),
                "music_present": int(row["music_present"]),
                "vp_panns_16k": submission.aggregate_head_predictions(
                    outputs["voice_present"][lower:upper], "voice_present"),
                "mp_panns_16k": submission.aggregate_head_predictions(
                    outputs["music_present"][lower:upper], "music_present"),
            })
        print(f"PANNs {min(start + batch_size, len(frame))}/{len(frame)}", flush=True)
    return pd.DataFrame(records), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["fusion_calibration"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_segments", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="experiments/panns_ab")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PANNsPresenceWrapper().to(device).eval()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "status": "MEASURED_LOCAL_VALIDATION",
            "frontend": OFFICIAL_16K_CONFIG.__dict__,
            "max_segments": args.max_segments,
            "splits": {},
        }
    for split in args.splits:
        split_path = ROOT / "data" / "splits" / f"{split}.csv"
        frame = balanced_order(pd.read_csv(split_path))
        predictions, elapsed = infer_split(model, frame, device, args.batch_size, args.max_segments)
        output_path = output_dir / f"{split}_16k.csv"
        predictions.to_csv(output_path, index=False)
        split_report: dict[str, object] = {
            "samples": len(predictions),
            "runtime_seconds": elapsed,
            "seconds_per_file": elapsed / max(1, len(predictions)),
            "corrected_16k": metrics(predictions, "vp_panns_16k", "mp_panns_16k"),
        }
        if split == "fusion_calibration":
            legacy_path = ROOT / "experiments" / "fusion_calibration_predictions.csv"
            legacy = pd.read_csv(legacy_path)
            keys = ["source", "generator", "voice_present", "music_present"]
            comparable = (
                legacy["source"].astype(str).tolist() == predictions["source"].tolist()
                and legacy["generator"].astype(str).tolist() == predictions["generator"].tolist()
                and legacy["y_voice_present"].astype(int).tolist() == predictions["voice_present"].tolist()
                and legacy["y_music_present"].astype(int).tolist() == predictions["music_present"].tolist()
            )
            if not comparable:
                raise RuntimeError(f"Calibration ordering mismatch; keys checked: {keys}")
            predictions["vp_panns_legacy"] = legacy["vp_panns"].to_numpy()
            predictions["mp_panns_legacy"] = legacy["mp_panns"].to_numpy()
            predictions["vp_model"] = legacy["vp_model"].to_numpy()
            predictions["mp_model"] = legacy["mp_model"].to_numpy()
            predictions.to_csv(output_path, index=False)
            split_report["legacy_mismatched"] = metrics(
                predictions, "vp_panns_legacy", "mp_panns_legacy")
            blend_results = {}
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                predictions["vp_blend"] = weight * predictions["vp_panns_16k"] + (1 - weight) * predictions["vp_model"]
                predictions["mp_blend"] = weight * predictions["mp_panns_16k"] + (1 - weight) * predictions["mp_model"]
                blend_results[str(weight)] = metrics(predictions, "vp_blend", "mp_blend")
            split_report["corrected_blend_grid"] = blend_results
        report["splits"][split] = split_report
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({split: split_report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
