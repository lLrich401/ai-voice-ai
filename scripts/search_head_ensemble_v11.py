#!/usr/bin/env python3
"""Cached head-ensemble search with robust domains and paired bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ensemble import (
    assert_final_holdout_forbidden, blend_probabilities,
    predict_head_selective_ensemble, validate_ensemble_cache_metadata,
)
from src.metrics import compute_dacon_metrics, compute_eer
from scripts.build_ensemble_cache_v11 import AGGREGATION_VERSION, SCHEMA_VERSION, sha256
import script as submission


DOMAINS = ("val_a", "val_b", "val_c", "val_d")
ALPHAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0)
METHODS = ("probability", "logit", "rank", "max")
HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def checkpoint_hashes() -> dict:
    paths = {
        "v7_voice": ROOT / "model/best.pt",
        "v7_music": ROOT / "model/music_best.pt",
        "v9_voice": ROOT / "model/candidates/voice_aasist_v9.pt",
        "v9_music": ROOT / "model/candidates/music_spec_cnn_v9.pt",
    }
    return {name: sha256(path) for name, path in paths.items()}


def load_cache(path: pathlib.Path, expected_split: pathlib.Path) -> pd.DataFrame:
    assert_final_holdout_forbidden(path, expected_split)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": submission.PIPELINE_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "checkpoint_sha256": checkpoint_hashes(),
        "split_sha256": sha256(expected_split),
    }
    validate_ensemble_cache_metadata(metadata, expected)
    frame = pd.read_csv(path)
    if len(frame) != int(metadata["rows"]):
        raise RuntimeError(f"cache row-count mismatch: {path}")
    return frame


def metric_from_predictions(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    truth = {head: frame[f"y_{head}"].to_numpy() for head in HEADS}
    scores = {head: predicted[:, index] for index, head in enumerate(HEADS)}
    return compute_dacon_metrics(truth, scores)


def config_predictions(frame: pd.DataFrame, weights: dict, config: dict) -> np.ndarray:
    voice = blend_probabilities(
        frame["vf"], frame["v9_vf"], config["voice_alpha"], config["voice_method"])
    music = blend_probabilities(
        frame["mf"], frame["v9_mf"], config["music_alpha"], config["music_method"])
    return predict_head_selective_ensemble(
        frame, weights, voice_fake=voice, music_fake=music,
        voice_affects_file=config["voice_affects_file"],
        music_affects_file=config["music_affects_file"],
        use_candidate_voice_file_head=False,
        use_candidate_music_file_head=False,
    )


def robust_summary(domain_metrics: dict) -> dict:
    mean_total = float(np.mean([domain_metrics[name]["total"] for name in DOMAINS]))
    worst_total = float(np.min([domain_metrics[name]["total"] for name in DOMAINS]))
    unseen_total = float(domain_metrics["val_b"]["total"])
    mean_ads = float(np.mean([domain_metrics[name]["ads"] for name in DOMAINS]))
    robust = 0.35 * mean_total + 0.25 * worst_total + 0.20 * unseen_total + 0.20 * mean_ads
    return {
        "mean_total": mean_total, "worst_total": worst_total,
        "unseen_total": unseen_total, "mean_ads": mean_ads,
        "robust_objective": float(robust),
        "mean_file_eer": float(np.mean([domain_metrics[name]["file_eer"] for name in DOMAINS])),
        "mean_voice_eer": float(np.mean([domain_metrics[name]["voice_eer"] for name in DOMAINS])),
        "worst_voice_eer": float(np.max([domain_metrics[name]["voice_eer"] for name in DOMAINS])),
        "mean_music_eer": float(np.mean([domain_metrics[name]["music_eer"] for name in DOMAINS])),
        "worst_music_eer": float(np.max([domain_metrics[name]["music_eer"] for name in DOMAINS])),
        "mean_cps": float(np.mean([domain_metrics[name]["cps"] for name in DOMAINS])),
    }


def unseen_eer(unseen: pd.DataFrame, config: dict, component: str) -> float:
    if component == "voice":
        rows = unseen[unseen["data_role"].eq("val_b_unseen_generator")]
        score = blend_probabilities(
            rows["v7_vf"], rows["v9_vf"], config["voice_alpha"], config["voice_method"])
        label = rows["y_voice_fake"]
    else:
        rows = unseen[unseen["data_role"].eq("val_b_unseen_music_generator")]
        score = blend_probabilities(
            rows["v7_mf"], rows["v9_mf"], config["music_alpha"], config["music_method"])
        label = rows["y_music_fake"]
    return compute_eer(label, score)


def evaluate_config(caches: dict, unseen: pd.DataFrame, weights: dict, config: dict) -> dict:
    metrics = {name: metric_from_predictions(frame, config_predictions(frame, weights, config))
               for name, frame in caches.items()}
    summary = robust_summary(metrics)
    voice_unseen = unseen_eer(unseen, config, "voice")
    music_unseen = unseen_eer(unseen, config, "music")
    voice_quality = (0.4 * (1.0 - summary["mean_voice_eer"])
                     + 0.3 * (1.0 - summary["worst_voice_eer"])
                     + 0.3 * (1.0 - voice_unseen))
    music_quality = (0.4 * (1.0 - summary["mean_music_eer"])
                     + 0.3 * (1.0 - summary["worst_music_eer"])
                     + 0.3 * (1.0 - music_unseen))
    return {
        "config": config, "domains": metrics, **summary,
        "voice_unseen_eer": float(voice_unseen),
        "music_unseen_eer": float(music_unseen),
        "voice_selection_objective": float(0.8 * summary["robust_objective"] + 0.2 * voice_quality),
        "music_selection_objective": float(0.8 * summary["robust_objective"] + 0.2 * music_quality),
        "joint_selection_objective": float(
            0.7 * summary["robust_objective"] + 0.15 * voice_quality + 0.15 * music_quality),
    }


def basic_config(**updates) -> dict:
    config = {
        "voice_alpha": 0.0, "voice_method": "probability",
        "music_alpha": 0.0, "music_method": "probability",
        "voice_affects_file": False, "music_affects_file": False,
        "deployable": True,
    }
    config.update(updates)
    return config


def eligible(result: dict, baseline: dict) -> bool:
    return (
        result["config"]["deployable"]
        and result["robust_objective"] > baseline["robust_objective"] + 1e-12
        and result["worst_total"] >= baseline["worst_total"] - 0.005
        and result["unseen_total"] >= baseline["unseen_total"] - 0.005
        and result["mean_file_eer"] <= baseline["mean_file_eer"] + 0.01
    )


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "domains"}


def bootstrap(caches: dict, weights: dict, baseline_config: dict,
              candidate_config: dict, iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    baseline_predictions = {
        name: config_predictions(frame, weights, baseline_config) for name, frame in caches.items()}
    candidate_predictions = {
        name: config_predictions(frame, weights, candidate_config) for name, frame in caches.items()}
    delta_names = ("file_eer", "voice_eer", "music_eer", "ads", "cps", "total")
    deltas = {name: [] for name in delta_names}
    robust_deltas = []
    for _ in range(iterations):
        base_metrics, candidate_metrics = {}, {}
        for name, frame in caches.items():
            indices = rng.integers(0, len(frame), len(frame))
            sampled = frame.iloc[indices].reset_index(drop=True)
            base_metrics[name] = metric_from_predictions(sampled, baseline_predictions[name][indices])
            candidate_metrics[name] = metric_from_predictions(sampled, candidate_predictions[name][indices])
        base_summary, candidate_summary = robust_summary(base_metrics), robust_summary(candidate_metrics)
        robust_deltas.append(candidate_summary["robust_objective"] - base_summary["robust_objective"])
        for metric in delta_names:
            deltas[metric].append(float(np.mean([
                candidate_metrics[name][metric] - base_metrics[name][metric] for name in DOMAINS])))
    report = {
        "iterations": iterations,
        "robust_win_rate": float(np.mean(np.asarray(robust_deltas) > 0.0)),
        "robust_delta": distribution(robust_deltas),
        "metric_delta_candidate_minus_v7": {
            name: distribution(values) for name, values in deltas.items()
        },
    }
    return report


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "win_rate": float(np.mean(values > 0.0)),
    }


def grid_rows(results: list) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {**result["config"]}
        for key in ("robust_objective", "joint_selection_objective", "mean_total",
                    "worst_total", "unseen_total", "mean_ads", "mean_file_eer",
                    "mean_voice_eer", "worst_voice_eer", "mean_music_eer",
                    "worst_music_eer", "voice_unseen_eer", "music_unseen_eer"):
            row[key] = result[key]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="experiments/v11/cache")
    parser.add_argument("--output", default="experiments/v11/ensemble_search.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    cache_dir = ROOT / args.cache_dir
    output = ROOT / args.output
    assert_final_holdout_forbidden(cache_dir, output)
    caches = {name: load_cache(
        cache_dir / f"original_{name}.csv", ROOT / f"data/splits/{name}.csv")
        for name in DOMAINS}
    unseen = load_cache(
        cache_dir / "expanded_unseen.csv", ROOT / "data/splits_v9_candidate/val_b.csv")
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    baseline_config = basic_config()
    baseline = evaluate_config(caches, unseen, weights, baseline_config)

    music_results = []
    voice_results = []
    for method in METHODS:
        alphas = (1.0,) if method == "max" else ALPHAS
        for alpha in alphas:
            deployable = method != "rank"
            for affects_file in (False, True):
                music_results.append(evaluate_config(caches, unseen, weights, basic_config(
                    music_alpha=alpha, music_method=method,
                    music_affects_file=affects_file, deployable=deployable)))
                voice_results.append(evaluate_config(caches, unseen, weights, basic_config(
                    voice_alpha=alpha, voice_method=method,
                    voice_affects_file=affects_file, deployable=deployable)))

    music_eligible = [item for item in music_results if eligible(item, baseline)]
    voice_eligible = [item for item in voice_results if eligible(item, baseline)]
    best_music = max(music_eligible, key=lambda item: item["music_selection_objective"], default=baseline)
    best_voice = max(voice_eligible, key=lambda item: item["voice_selection_objective"], default=baseline)

    # Explore only a bounded cross-product of independently robust candidates.
    top_music = sorted(music_eligible, key=lambda item: item["music_selection_objective"], reverse=True)[:5]
    top_voice = sorted(voice_eligible, key=lambda item: item["voice_selection_objective"], reverse=True)[:5]
    joint_results = []
    for voice_result in top_voice or [baseline]:
        for music_result in top_music or [baseline]:
            config = dict(voice_result["config"])
            for key in ("music_alpha", "music_method", "music_affects_file"):
                config[key] = music_result["config"][key]
            config["deployable"] = (voice_result["config"]["deployable"]
                                    and music_result["config"]["deployable"])
            joint_results.append(evaluate_config(caches, unseen, weights, config))
    joint_eligible = [item for item in joint_results if eligible(item, baseline)]
    best_joint = max(joint_eligible, key=lambda item: item["joint_selection_objective"], default=baseline)

    selected = {"B_music": best_music, "C_voice": best_voice, "D_joint": best_joint}
    bootstrap_reports = {
        name: bootstrap(caches, weights, baseline_config, result["config"], args.bootstrap,
                        20260903 + index)
        for index, (name, result) in enumerate(selected.items())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    grid_rows(music_results).to_csv(output.parent / "music_grid.csv", index=False)
    grid_rows(voice_results).to_csv(output.parent / "voice_grid.csv", index=False)
    grid_rows(joint_results).to_csv(output.parent / "joint_grid.csv", index=False)
    report = {
        "status": "MEASURED_NON_FINAL_CACHED_VALIDATION",
        "final_holdout": "NOT RUN",
        "objective": "0.35*mean_TOTAL + 0.25*worst_TOTAL + 0.20*VAL-B_TOTAL + 0.20*mean_ADS",
        "baseline": baseline,
        "best_music": best_music,
        "best_voice": best_voice,
        "best_joint": best_joint,
        "bootstrap": bootstrap_reports,
        "grid_counts": {
            "music": len(music_results), "voice": len(voice_results),
            "joint": len(joint_results), "bootstrap_iterations": args.bootstrap,
        },
        "rank_note": "measured but ineligible because batch-rank output violates independent-file inference",
        "candidate_file_heads_used": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": compact(baseline), "best_music": compact(best_music),
        "best_voice": compact(best_voice), "best_joint": compact(best_joint),
        "bootstrap": bootstrap_reports,
    }, indent=2))


if __name__ == "__main__":
    main()
