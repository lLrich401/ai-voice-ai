#!/usr/bin/env python3
"""Compare frozen TEST5 and current x1 inference on identical validation audio."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script as current
from tools.v13_guards import assert_final_holdout_v13b_forbidden


TOLERANCE = 1e-6


def load_frozen_script():
    archive = ROOT / "archive/pre_v13_selected/submit.zip"
    with zipfile.ZipFile(archive) as handle:
        source = handle.read("script.py")
    temporary = tempfile.NamedTemporaryFile(suffix="_frozen_test5.py", delete=False)
    temporary.write(source); temporary.close()
    path = pathlib.Path(temporary.name)
    spec = importlib.util.spec_from_file_location("frozen_test5_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def rows_to_array(rows: list[list]) -> tuple[list[str], np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(row[0]))
    return [str(row[0]) for row in ordered], np.asarray([row[1:] for row in ordered], dtype=float)


def run(module, models: tuple, weights: dict, paths: list[pathlib.Path],
        batch: int, checkpoints: tuple[int, ...]) -> tuple[list[list], dict[int, dict]]:
    voice, music, df, panns, device = models
    started = time.perf_counter(); result = []; batch_rates = []; snapshots = {}
    for start in range(0, len(paths), batch):
        batch_started = time.perf_counter()
        result.extend(module.infer_files_batch(
            voice, music, df, panns, weights, paths[start:start + batch], device,
            use_demucs=False))
        completed = min(start + batch, len(paths))
        batch_count = completed - start
        batch_rates.append((time.perf_counter() - batch_started) / batch_count)
        if completed in checkpoints:
            total = time.perf_counter() - started
            snapshots[completed] = {
                "total_seconds": total,
                "seconds_per_file": total / completed,
                "median_batch_seconds_per_file": float(np.median(batch_rates)),
                "files_per_second": completed / total,
                "projected_1200_file_minutes": total / completed * 1200 / 60,
            }
    return result, snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=(64, 128))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", default="experiments/v13b/end_to_end_runtime_benchmark.json")
    args = parser.parse_args()
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if not checkpoints or checkpoints[-1] != args.samples:
        raise ValueError("largest --checkpoints value must equal --samples")
    if any(value <= 0 or value % args.batch for value in checkpoints):
        raise ValueError("runtime checkpoints must be positive multiples of batch size")
    split = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
    assert_final_holdout_v13b_forbidden(split, args.output)
    frame = pd.read_csv(split)
    direct = frame[~frame.path.astype(str).str.startswith(("MIX::", "PARTIAL::"))]
    paths = [pathlib.Path(value) for value in direct.path.iloc[:args.samples]]
    if len(paths) != args.samples or not all(path.is_file() for path in paths):
        raise RuntimeError("runtime benchmark requires the requested local direct validation files")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current.verify_mandatory_models()
    models = (current.load_voice_model(device), current.load_music_model(device),
              current.load_df_arena(device), current.load_panns(device), device)
    weights = current.load_fusion_weights()
    frozen, temporary = load_frozen_script()
    try:
        # Excluded warmup removes model/JIT initialization from both timings.
        current.infer_files_batch(*models[:4], weights, paths[:1], device, use_demucs=False)
        frozen_rows, frozen_scales = run(frozen, models, weights, paths, args.batch, checkpoints)
        current_rows, current_scales = run(current, models, weights, paths, args.batch, checkpoints)
    finally:
        temporary.unlink(missing_ok=True)
    frozen_ids, frozen_values = rows_to_array(frozen_rows)
    current_ids, current_values = rows_to_array(current_rows)
    if frozen_ids != current_ids:
        raise RuntimeError("TEST5/x1 runtime benchmark ID mismatch")
    maximum_difference = float(np.max(np.abs(frozen_values - current_values)))
    parity = maximum_difference <= TOLERANCE
    report = {
        "status": "PASS" if parity else "FAIL_PREDICTION_PARITY",
        "scope": "MEASURED_LOCAL generator-disjoint direct subset; linear projection is PROJECTED",
        "device": device, "files": len(paths), "batch": args.batch,
        "checkpoints": list(checkpoints),
        "predefined_prediction_tolerance": TOLERANCE,
        "decode_strategy": "per-batch executor; persistent executor rejected after no measured gain",
        "prediction_max_abs_diff": maximum_difference, "prediction_parity": parity,
        "scales": {
            str(size): {
                "frozen_test5": frozen_scales[size],
                "current_x1": current_scales[size],
                "relative_speedup": (
                    frozen_scales[size]["seconds_per_file"] /
                    current_scales[size]["seconds_per_file"]),
            }
            for size in checkpoints
        },
        "frozen_test5": frozen_scales[args.samples],
        "current_x1": current_scales[args.samples],
        "official_test5_runtime": "30m52s USER_REPORTED_PUBLIC",
        "official_x1_runtime": "NOT RUN",
        "final_holdout": "NOT READ / NOT RUN",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not parity:
        raise RuntimeError(f"prediction parity failed: {maximum_difference} > {TOLERANCE}")


if __name__ == "__main__":
    main()
