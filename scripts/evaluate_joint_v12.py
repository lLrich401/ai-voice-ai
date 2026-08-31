#!/usr/bin/env python3
"""Evaluate the preselected M4+V2 V12 student pair without final holdout access."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_students_v12 import VALS, replace_student, robust_fold_summary, unseen_metrics
from src.ensemble import assert_final_holdout_forbidden, score_head_selective_ensemble


def load_joint(name: str) -> pd.DataFrame:
    base = pd.read_csv(ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv")
    music_path = ROOT / f"experiments/v12/cache/music_students/m4/{name}.csv"
    voice_path = ROOT / f"experiments/v12/cache/voice_students/v2/{name}.csv"
    result = replace_student(base, "music", music_path)
    return replace_student(result, "voice", voice_path)


def summarize(weights: dict) -> dict:
    domains = {name: score_head_selective_ensemble(load_joint(name), weights) for name in VALS}
    expanded = load_joint("expanded_unseen")
    unseen = unseen_metrics(expanded, weights)
    calibration = {
        name: robust_fold_summary(load_joint(name), weights)
        for name in ("cal_old", "cal_v12")
    }
    totals = np.asarray([value["total"] for value in domains.values()], dtype=float)
    mean = {
        key: float(np.mean([value[key] for value in domains.values()]))
        for key in ("file_eer", "voice_eer", "music_eer", "ads", "cps", "total")
    }
    robust = (0.25 * totals.mean() + 0.15 * totals.min()
              + 0.15 * (1.0 - unseen["voice_eer"])
              + 0.15 * (1.0 - unseen["music_eer"])
              + 0.15 * calibration["cal_old"]["robust"]
              + 0.15 * calibration["cal_v12"]["robust"])
    return {
        "candidate": "M4_music_plus_V2_voice",
        "domains": domains,
        "mean": mean,
        "worst_total": float(totals.min()),
        "unseen": unseen,
        "calibration": calibration,
        "robust_objective": float(robust),
        "final_holdout": "NOT RUN",
    }


def main() -> None:
    output = ROOT / "experiments/v12/joint_evaluation.json"
    assert_final_holdout_forbidden(output)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    report = summarize(weights)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
