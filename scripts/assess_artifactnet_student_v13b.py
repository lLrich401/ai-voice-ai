#!/usr/bin/env python3
"""Fail-closed direct-versus-student feasibility decision for ArtifactNet.

This is deliberately an approval gate, not a training launcher.  A student
would inherit a teacher's usage/provenance constraints, so no distillation
labels are exported until the teacher is allowed for the competition and its
direct numerical path has passed the required non-final domains.
"""

from __future__ import annotations

import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "experiments/v13b/artifactnet_candidate_provenance.json"
CAL = ROOT / "experiments/v13b/artifactnet_calibration.json"
GENERATOR = ROOT / "experiments/v13b/artifactnet_generator_confirmation.json"
NONFINITE = ROOT / "experiments/v13b/artifactnet_nonfinite_analysis.json"
OUTPUT = ROOT / "experiments/v13b/artifactnet_direct_vs_student.json"


def main() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    generator = json.loads(GENERATOR.read_text(encoding="utf-8"))
    nonfinite = json.loads(NONFINITE.read_text(encoding="utf-8"))
    licence_ok = (provenance.get("competition_use_approval") == "APPROVED" and
                  provenance.get("redistribution_in_submission") == "APPROVED")
    # The selected direct policy is one high-energy crop.  Multi-crop is a
    # rejected alternative, so its NaN cannot be misreported as a failure of
    # the one-crop path; it remains a hidden-domain stability warning.
    direct_finite = (cal.get("frontend", {}).get("selected") is not None and
                     generator.get("status") == "PASS_CONFIRMATION" and
                     int(generator.get("nonfinite_segment_count", 0)) == 0)
    direct = {
        "status": "ELIGIBLE_FOR_FURTHER_RESEARCH" if licence_ok and direct_finite else "REJECTED_FAIL_CLOSED",
        "license_approval": licence_ok,
        "selected_one_crop_numerical_stability": direct_finite,
        "multi_crop_policy": ("REJECTED_NONFINITE" if int(nonfinite.get("nonfinite_count", 0)) else
                              "NOT_MEASURED"),
        "calibration_status": cal.get("status"),
        "generator_confirmation_status": generator.get("status"),
        "post_policy_diagnostic_nonfinite_count": nonfinite.get("nonfinite_count"),
    }
    student = {
        "status": "BLOCKED_BEFORE_TRAINING",
        "reason": [
            "teacher competition-use and redistribution approval are not confirmed",
            "teacher is not eligible for a competition submission even though selected one-crop diagnostics passed",
            "no teacher scores or labels may be exported from CAL/generator validation",
        ],
        "permitted_after_all_gates": [
            "generate teacher labels from train_v13b only",
            "train student from train_v13b only",
            "use CAL for one policy decision and generator-disjoint only for confirmation",
            "retain teacher/student provenance and licence chain in candidate metadata",
        ],
    }
    payload = {
        "status": "BLOCKED" if student["status"] != "ELIGIBLE" else "PASS",
        "direct": direct, "student": student,
        "selected_artifacts_mutated": False, "final_holdout": "NOT RUN",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
