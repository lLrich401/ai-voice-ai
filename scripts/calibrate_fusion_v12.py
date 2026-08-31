#!/usr/bin/env python3
"""Conservative CAL_OLD/CAL_V12 cross-generalization for the best V12 point candidate."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_students_v12 import VALS, replace_student
from src.ensemble import assert_final_holdout_forbidden, score_head_selective_ensemble


def fold_summary(frame: pd.DataFrame, weights: dict) -> dict:
    folds = {str(name): score_head_selective_ensemble(group, weights)
             for name, group in frame.groupby("calibration_fold", sort=True)}
    totals = np.asarray([value["total"] for value in folds.values()])
    return {"folds": folds, "mean": float(totals.mean()),
            "worst": float(totals.min()),
            "robust": float(0.7 * totals.mean() + 0.3 * totals.min())}


def val_summary(frames: dict[str, pd.DataFrame], weights: dict) -> dict:
    domains = {name: score_head_selective_ensemble(frame, weights)
               for name, frame in frames.items()}
    totals = np.asarray([value["total"] for value in domains.values()])
    return {"domains": domains, "mean": float(totals.mean()),
            "worst": float(totals.min())}


def fit_score(old: dict, new: dict, mode: str) -> float:
    if mode == "old":
        return old["robust"]
    if mode == "v12":
        return new["robust"]
    return float(0.35 * old["mean"] + 0.35 * new["mean"]
                 + 0.15 * old["worst"] + 0.15 * new["worst"])


def coordinate_fit(initial: dict, old_frame: pd.DataFrame, new_frame: pd.DataFrame,
                   mode: str) -> dict:
    current = dict(initial)
    detector = [(0.0, 0.5, 0.5), (0.25, 0.5, 0.25), (0.0, 0.0, 1.0)]
    coordinates = (
        ("detector", detector),
        ("w_df_arena", [0.0, 0.25, 0.5]),
        ("w_df_voice_component", [0.2, 0.3, 0.35]),
        ("w_df_music_component", [0.0, 0.05]),
        ("file_fusion_mode", ["legacy", "presence_weighted", "presence_component_or"]),
    )
    trace = []
    for pass_index in range(2):
        for name, values in coordinates:
            choices = []
            for value in values:
                trial = dict(current)
                if name == "detector":
                    trial.update(dict(zip(
                        ("w_voice_file", "w_music_file", "w_prob_or"), value)))
                else:
                    trial[name] = value
                old = fold_summary(old_frame, trial)
                new = fold_summary(new_frame, trial)
                score = fit_score(old, new, mode)
                distance = (abs(float(trial["w_df_arena"]) - float(initial["w_df_arena"]))
                            + abs(float(trial["w_df_voice_component"])
                                  - float(initial["w_df_voice_component"])))
                choices.append((score, -distance, trial, old, new))
            score, _, current, old, new = max(choices, key=lambda item: item[:2])
            trace.append({"pass": pass_index + 1, "coordinate": name,
                          "score": score, "weights": dict(current)})
    return {"fit": mode, "weights": current,
            "cal_old": fold_summary(old_frame, current),
            "cal_v12": fold_summary(new_frame, current),
            "fit_objective": fit_score(fold_summary(old_frame, current),
                                       fold_summary(new_frame, current), mode),
            "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-candidate", default="V2")
    parser.add_argument("--music-candidate", default="M4")
    args = parser.parse_args()
    voice_candidate = args.voice_candidate.upper()
    music_candidate = args.music_candidate.upper()
    output = ROOT / "experiments/v12/fusion_cross_generalization.json"
    assert_final_holdout_forbidden(output)
    stored_weights = json.loads(
        (ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    # Keep the experiment artifact focused on the actual searched parameters;
    # the selected fusion JSON remains the immutable source for provenance.
    keys = ("w_voice_file", "w_music_file", "w_prob_or", "w_df_arena",
            "w_df_voice_component", "w_df_music_component", "w_panns_presence",
            "file_fusion_mode")
    base_weights = {key: stored_weights[key] for key in keys}
    frames = {}
    for name in (*VALS, "cal_old", "cal_v12"):
        base = pd.read_csv(ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv")
        with_music = replace_student(base, "music", ROOT /
            f"experiments/v12/cache/music_students/{music_candidate.lower()}/{name}.csv")
        frames[name] = replace_student(with_music, "voice", ROOT /
            f"experiments/v12/cache/voice_students/{voice_candidate.lower()}/{name}.csv")
    base_frames = {name: pd.read_csv(
        ROOT / f"experiments/v12/cache/v7_canonical/{name}.csv")
        for name in (*VALS, "cal_old", "cal_v12")}
    fits = {}
    for mode in ("old", "v12", "both"):
        result = coordinate_fit(base_weights, frames["cal_old"], frames["cal_v12"], mode)
        result["validation"] = val_summary({name: frames[name] for name in VALS}, result["weights"])
        fits[mode] = result
    baseline = {
        "cal_old": fold_summary(base_frames["cal_old"], base_weights),
        "cal_v12": fold_summary(base_frames["cal_v12"], base_weights),
        "validation": val_summary({name: base_frames[name] for name in VALS}, base_weights),
    }
    candidate_current_fusion = {
        "cal_old": fold_summary(frames["cal_old"], base_weights),
        "cal_v12": fold_summary(frames["cal_v12"], base_weights),
        "validation": val_summary({name: frames[name] for name in VALS}, base_weights),
    }
    report = {
        "status": "MEASURED_CONSERVATIVE_COORDINATE_SEARCH",
        "candidate": f"{music_candidate}_music_plus_{voice_candidate}_voice",
        "precommitted_calibration_both_objective": (
            "0.35 CAL_OLD mean + 0.35 CAL_V12 mean + 0.15 CAL_OLD worst + 0.15 CAL_V12 worst"),
        "baseline": baseline,
        "candidate_current_fusion": candidate_current_fusion,
        "fits": fits,
        "cross_generalization": {
            "fit_CAL_OLD_evaluate_CAL_V12": fits["old"]["cal_v12"],
            "fit_CAL_V12_evaluate_CAL_OLD": fits["v12"]["cal_old"],
            "fit_CAL_OLD_evaluate_VAL": fits["old"]["validation"],
            "fit_CAL_V12_evaluate_VAL": fits["v12"]["validation"],
            "fit_BOTH_fold_safe_proxy": {
                "cal_old": fits["both"]["cal_old"],
                "cal_v12": fits["both"]["cal_v12"],
                "validation": fits["both"]["validation"],
            },
        },
        "logistic_file_meta": (
            "NOT RUN: the request permits logistic file meta only after stable component "
            "selection; conservative coordinate fusion is sufficient for this iteration"),
        "final_holdout": "NOT RUN",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    concise = {mode: {"weights": value["weights"],
                      "old": value["cal_old"]["robust"],
                      "v12": value["cal_v12"]["robust"],
                      "val_mean": value["validation"]["mean"],
                      "val_worst": value["validation"]["worst"]}
               for mode, value in fits.items()}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
