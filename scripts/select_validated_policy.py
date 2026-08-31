#!/usr/bin/env python3
"""Apply a gate policy only when it does not regress any validation domain."""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_fusion import robust_score
REPORT = ROOT / "experiments/validation_domain_gate_report.json"
WEIGHTS = ROOT / "model/fusion_weights.json"


def robust(values):
    return 0.7 * sum(values) / len(values) + 0.3 * min(values)


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    current = []
    selected = []
    adaptive = []
    per_domain = {}
    for split, payload in report["splits"].items():
        policies = payload["policies"]
        old = policies["A_current_voice_0.8"]["metrics"]["total"]
        primary = policies["B_full_primary"]["metrics"]["total"]
        second = policies["C_full_adaptive"]["metrics"]["total"]
        current.append(old)
        selected.append(primary)
        adaptive.append(second)
        per_domain[split] = {"current": old, "full_primary": primary,
                             "full_adaptive": second,
                             "full_primary_delta": primary - old}
    tolerance = 1e-12
    if any(new + tolerance < old for new, old in zip(selected, current)):
        raise RuntimeError("Full-primary DF regresses at least one validation domain")
    weights = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    weights.update({
        "df_gate_policy": "off",
        "df_gate_calibration_fraction": 1.0,
        "adaptive_df_enabled": False,
        "voice_fake_aggregation": "max",
        "policy_selection": {
            "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
            "selected": "full_primary_df_gate_off",
            "adaptive_rejected_reason": "VAL-C codec TOTAL regressed by 0.00375",
            "current_domain_robust": robust(current),
            "selected_domain_robust": robust(selected),
            "adaptive_domain_robust": robust(adaptive),
            "per_domain": per_domain,
            "report": "experiments/validation_domain_gate_report.json",
            "voice_aggregation": {
                "selected": "max",
                "report": "experiments/voice_aggregation_report.json",
                "reason": "lower raw VOICE EER on VAL-A/B/C/D and no fused TOTAL regression",
            },
        },
    })
    weights.pop("df_gate_voice_presence_threshold", None)
    calibration = pd.read_csv(
        ROOT / "experiments/fusion_calibration_predictions_16k_voice_max.csv")
    no_adaptive = {"enabled": False, "low": 0.0, "high": 1.0,
                   "aggregation": "mean"}
    objective, fold_metrics = robust_score(calibration, weights, no_adaptive)
    weights["calibration_robust_objective"] = objective
    weights["calibration_metrics_by_fold"] = fold_metrics
    WEIGHTS.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(json.dumps(weights["policy_selection"], indent=2))


if __name__ == "__main__":
    main()
