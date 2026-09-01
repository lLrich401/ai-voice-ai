#!/usr/bin/env python3
"""Repeat an already-observed ArtifactNet non-finite segment without tuning.

This diagnostic is intentionally post-selection: it consumes the frozen policy
and only the immutable row already recorded as non-finite.  It cannot change a
frontend, gate, aggregate, checkpoint, or submission decision.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research_artifactnet_v13b import (
    CEILINGS, OUTPUT_NONFINITE, ceiling_name, selected_segments, session,
)
from scripts.evaluate_v13b_artifactnet import to_artifactnet_input
from tools.v13_guards import assert_final_holdout_v13b_forbidden


OUTPUT = ROOT / "experiments/v13b/artifactnet_nonfinite_reproduction.json"
REPEATS_SAME_SESSION = 8
REPEATS_FRESH_SESSION = 4


def finite_output(value: np.ndarray) -> bool:
    return value.size == 1 and bool(np.isfinite(value).all()) and 0.0 <= float(value[0]) <= 1.0


def run_once(sess, audio: np.ndarray) -> tuple[bool, list[float]]:
    value = np.asarray(sess.run(None, {"audio": audio.reshape(1, -1)})[0],
                       dtype=np.float64).reshape(-1)
    return finite_output(value), value.tolist()


def main() -> None:
    assert_final_holdout_v13b_forbidden(OUTPUT_NONFINITE, OUTPUT)
    report = json.loads(OUTPUT_NONFINITE.read_text(encoding="utf-8"))
    records = report.get("records", [])
    payload = {
        "status": "MEASURED" if records else "NOT_RUN_NO_RECORDED_NONFINITE_SEGMENT",
        "policy_frozen_from_cal": report.get("policy_frozen_from_cal"),
        "records": [], "selection_effect": "NONE; diagnostic only",
        "final_holdout": "NOT RUN",
    }
    for record in records:
        row = {"source_path": record["source_path"], "path": record["path"]}
        segment = selected_segments(row, int(record["segment_index"]) + 1)[int(record["segment_index"])]
        ceiling = record.get("ceiling")
        audio, _ = to_artifactnet_input(segment, ceiling)
        same = session()
        same_results = [run_once(same, audio) for _ in range(REPEATS_SAME_SESSION)]
        fresh_results = [run_once(session(), audio) for _ in range(REPEATS_FRESH_SESSION)]
        policy_trials = []
        for trial_ceiling in CEILINGS:
            trial, _ = to_artifactnet_input(segment, trial_ceiling)
            finite, output = run_once(session(), trial)
            policy_trials.append({"policy": ceiling_name(trial_ceiling), "finite": finite,
                                  "output": output})
        payload["records"].append({
            "audio_id": record["audio_id"], "path": record["path"],
            "segment_index": record["segment_index"], "same_session": same_results,
            "fresh_sessions": fresh_results, "all_same_session_finite": all(value[0] for value in same_results),
            "all_fresh_session_finite": all(value[0] for value in fresh_results),
            "frontend_trials": policy_trials,
        })
    payload["interpretation"] = (
        "A finite repeat does not clear the original failure; production remains fail-closed. "
        "It only distinguishes a reproducible input pathology from an intermittent ONNX/session-path failure.")
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
