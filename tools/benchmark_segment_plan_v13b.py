#!/usr/bin/env python3
"""Measure identity-separator segment planning reuse without model inference."""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import script


OUT = ROOT / "experiments/v13b/segment_plan_benchmark.json"


def old_three_scans(wave: np.ndarray) -> tuple[list[np.ndarray], ...]:
    return tuple(script.limit_aux_segments(script.select_aux_segments(wave), 3)
                 for _ in range(3))


def one_reused_scan(wave: np.ndarray) -> tuple[list[np.ndarray], ...]:
    plan = script.build_segment_plan(wave)
    duration = len(wave) / 16000.0
    selected = script.select_segments_from_plan(plan, duration, "high_energy", 3)
    return selected, selected, selected


def main() -> None:
    generator = np.random.default_rng(23674913)
    waves = [generator.normal(0, 0.1, seconds * 16000).astype(np.float32)
             for seconds in (7, 12, 20, 30, 40, 55)]
    repeats = 5
    measurements = {}
    outputs = {}
    for name, function in (("before_three_scans", old_three_scans),
                           ("after_one_reused_scan", one_reused_scan)):
        start = time.perf_counter()
        for _ in range(repeats):
            outputs[name] = [function(wave) for wave in waves]
        measurements[name] = time.perf_counter() - start
    parity = all(
        np.array_equal(left_segment, right_segment)
        for left_groups, right_groups in zip(outputs["before_three_scans"],
                                              outputs["after_one_reused_scan"])
        for left, right in zip(left_groups, right_groups)
        for left_segment, right_segment in zip(left, right)
    )
    before = measurements["before_three_scans"]
    after = measurements["after_one_reused_scan"]
    report = {
        "status": "ADOPTED" if parity and after < before else "REJECTED",
        "scope": "MEASURED_LOCAL segment extraction/energy planning only; model inference NOT RUN",
        "waves": len(waves), "repeats": repeats,
        "segment_scans_before_per_file": 3,
        "segment_scans_after_per_file": 1,
        "prediction_input_parity": parity,
        "before_seconds": before,
        "after_seconds": after,
        "speedup": before / after,
        "decode_executor": "per-batch executor retained; persistent executor rejected by measured end-to-end benchmark",
        "df_gpu_overlap": "NOT RUN: no eligible CUDA benchmark in this run",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
