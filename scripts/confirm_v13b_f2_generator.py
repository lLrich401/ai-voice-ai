#!/usr/bin/env python3
"""One-shot generator-disjoint confirmation of the CAL-frozen F2 weight."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v13b_file_complementarity import complementarity
from src.metrics import compute_eer
from tools.v13_guards import assert_final_holdout_v13b_forbidden


WEIGHT = 0.35  # Frozen by CAL-only selection; do not add a generator-val grid.
SPLIT = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
CACHE = ROOT / "experiments/v13b/m2_file_complementarity_scores.csv"
RUNTIME = ROOT / "experiments/v13b/m2_file_complementarity.json"
OUTPUT = ROOT / "experiments/v13b/f2_generator_confirmation.json"


def main() -> None:
    assert_final_holdout_v13b_forbidden(SPLIT, CACHE, RUNTIME, OUTPUT)
    frame = pd.read_csv(SPLIT)
    cached = pd.read_csv(CACHE)
    expected_ids = [pathlib.Path(value).stem for value in frame.path]
    if cached.audio_id.astype(str).tolist() != expected_ids:
        raise RuntimeError("F2 generator cache is not an exact split-ID match")
    required = {"file_fake", "canonical_test5_file_probability", "m2_file_probability"}
    if not required <= set(cached.columns):
        raise RuntimeError(f"F2 generator cache missing {sorted(required - set(cached.columns))}")
    truth = frame.file_fake.to_numpy(int)
    if not np.array_equal(truth, cached.file_fake.to_numpy(int)):
        raise RuntimeError("F2 generator cache labels do not match frozen split")
    canonical = cached.canonical_test5_file_probability.to_numpy(float)
    f2 = cached.m2_file_probability.to_numpy(float)
    blend = (1.0 - WEIGHT) * canonical + WEIGHT * f2
    if not np.isfinite(blend).all():
        raise RuntimeError("F2 generator blend is non-finite")
    baseline_eer = compute_eer(truth, canonical)
    blend_eer = compute_eer(truth, blend)
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    payload = {
        "status": "F2_GENERATOR_SIGNAL_ONLY" if blend_eer < baseline_eer else "F2_REJECT_AFTER_INDEPENDENT_CONFIRMATION",
        "selection": {"data": "cal_v13b only", "f2_weight_frozen": WEIGHT,
                      "generator_weight_search": "FORBIDDEN_NOT_RUN"},
        "rows": len(frame), "canonical_file_eer": baseline_eer, "f2_blend_file_eer": blend_eer,
        "delta_file_eer": baseline_eer - blend_eer,
        "delta_ads_file_component": 0.5 * (baseline_eer - blend_eer),
        "error_overlap": complementarity(truth, canonical, blend),
        "partial_file": "NOT RUN: frozen canonical virtual-partial predictions are not cached",
        "mixed_file": "NOT RUN: frozen canonical virtual-mixed predictions are not cached",
        "runtime": {
            "status": "MEASURED_COMPONENT_CACHE", "f2_added_1200_file_minutes_projected": runtime[
                "runtime"]["candidate_added_1200_file_minutes_projected"],
            "cache_inference_rerun": "NOT RUN; exact frozen score cache reused",
        },
        "source_disjoint": "NOT MEASURED", "bootstrap": "NOT RUN", "final_holdout": "NOT RUN",
        "selected_artifacts_mutated": False, "decision": "KEEP_TEST5",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
