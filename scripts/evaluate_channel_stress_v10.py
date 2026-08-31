#!/usr/bin/env python3
"""Evaluate deterministic codec/telephone stress profiles on unused VAL-A rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import balanced_subset, collect, score_frame
import script as submission


PROFILES = (
    ("clean", "none"),
    ("codec_lp35", "codec_lp35"),
    ("codec_lp52", "codec_lp52"),
    ("codec_resample12_q12", "codec_resample12_q12"),
    ("codec_narrow_q8", "codec_narrow_q8"),
    ("codec_wide_q10", "codec_wide_q10"),
    ("telephone_mulaw", "telephone_mulaw"),
    ("telephone_alaw", "telephone_alaw"),
    ("telephone_narrow", "telephone_narrow"),
    ("telephone_gsm_proxy", "telephone_gsm_proxy"),
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else
                                              "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                                              else "cpu"))
    parser.add_argument("--output", default="experiments/v10/channel_stress_report.json")
    parser.add_argument(
        "--profiles", default="all",
        help="comma-separated profile names, or all",
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    # ``script`` prepends the bundled runtime directory to sys.path. Importing
    # it before the calibration helpers would silently select that stale copy
    # of src.dataset and ignore newly added stress profiles.
    dataset_module = sys.modules[collect.__globals__["load_row_wave"].__globals__["load_manifest_row_wave"].__module__]
    expected_dataset = (ROOT / "src/dataset.py").resolve()
    actual_dataset = pathlib.Path(dataset_module.__file__).resolve()
    if actual_dataset != expected_dataset:
        raise RuntimeError(
            f"channel stress must use working source {expected_dataset}, got {actual_dataset}")
    source_path = ROOT / "data/splits/val_a.csv"
    source = pd.read_csv(source_path)
    # Keep the probe content separate from the historical 128-row A/B/C/D
    # caches used to identify the issue. This is still validation, never TRAIN.
    used = set(pd.read_csv(ROOT / "experiments/val_a_voice_aggregation.csv")["path"].astype(str))
    unused = source[~source["path"].astype(str).isin(used)].reset_index(drop=True)
    frame = balanced_subset(unused, args.samples, 20260902)
    if set(frame["path"].astype(str)) & used:
        raise RuntimeError("stress confirmation rows overlap the diagnostic cache")
    device = torch.device(args.device)
    models = (submission.load_voice_model(device), submission.load_music_model(device))
    panns = submission.load_panns(device)
    df_session = submission.load_df_arena(device)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    adaptive = {"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}
    report = {
        "status": "MEASURED_NON_FINAL_CHANNEL_STRESS",
        "final_holdout": "NOT RUN", "source_split": str(source_path),
        "source_split_sha256": sha256(source_path), "diagnostic_cache_overlap": 0,
        "samples_per_profile": int(len(frame)), "profiles": {},
    }
    requested = {value.strip() for value in args.profiles.split(",") if value.strip()}
    profiles = PROFILES if requested == {"all"} else tuple(
        profile for profile in PROFILES if profile[0] in requested)
    missing = requested - {name for name, _ in PROFILES}
    if missing and requested != {"all"}:
        raise ValueError(f"unknown profiles: {sorted(missing)}")
    for name, augment in profiles:
        candidate = frame.copy()
        candidate["augment"] = augment
        started = time.perf_counter()
        cache = pd.DataFrame(collect(
            name, candidate, models, df_session, panns, device, args.batch_size))
        metrics = score_frame(cache, weights, adaptive)
        cache.to_csv(output.parent / f"stress_{name}.csv", index=False)
        report["profiles"][name] = {
            "augment": augment, "runtime_seconds": time.perf_counter() - started,
            "metrics": metrics,
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(name, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
