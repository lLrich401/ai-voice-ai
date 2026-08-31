#!/usr/bin/env python3
"""Measure procedural-source fingerprint and sampling-balance risk."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import load_audio  # noqa: E402
from src.train import specialist_sample_weights  # noqa: E402


def acoustic_features(path: str, max_seconds: float = 8.0) -> dict[str, float]:
    wave, sample_rate = load_audio(str(ROOT / path) if not pathlib.Path(path).is_absolute() else path,
                                   target_sr=16_000)
    full_duration = len(wave) / sample_rate
    maximum = int(max_seconds * sample_rate)
    if len(wave) > maximum:
        start = (len(wave) - maximum) // 2
        wave = wave[start:start + maximum]
    wave = np.asarray(wave, dtype=np.float64)
    rms = float(np.sqrt(np.mean(wave ** 2) + 1e-12))
    peak = float(np.max(np.abs(wave)))
    zcr = float(np.mean(np.signbit(wave[1:]) != np.signbit(wave[:-1])))
    _, _, spectrum = stft(wave, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    power = np.abs(spectrum) ** 2 + 1e-12
    mean_power = power.mean(axis=1)
    frequencies = np.linspace(0.0, sample_rate / 2.0, len(mean_power))
    total = float(mean_power.sum())
    centroid = float(np.sum(frequencies * mean_power) / total)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * mean_power) / total))
    cumulative = np.cumsum(mean_power) / total
    rolloff85 = float(frequencies[min(len(frequencies) - 1, int(np.searchsorted(cumulative, 0.85)))])
    flatness = float(np.exp(np.mean(np.log(mean_power))) / np.mean(mean_power))
    frame_rms = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0) + 1e-12)
    result = {
        "duration": float(full_duration), "rms": rms, "peak": peak,
        "crest": peak / max(rms, 1e-8), "zcr": zcr,
        "abs_q10": float(np.quantile(np.abs(wave), 0.10)),
        "abs_q50": float(np.quantile(np.abs(wave), 0.50)),
        "abs_q90": float(np.quantile(np.abs(wave), 0.90)),
        "silence_fraction": float(np.mean(np.abs(wave) < 1e-3)),
        "clipping_fraction": float(np.mean(np.abs(wave) >= 0.98)),
        "spectral_centroid": centroid, "spectral_bandwidth": bandwidth,
        "spectral_rolloff85": rolloff85, "spectral_flatness": flatness,
        "frame_rms_std": float(np.std(frame_rms)),
        "frame_rms_p90_p10": float(np.quantile(frame_rms, 0.9) / max(np.quantile(frame_rms, 0.1), 1e-8)),
    }
    band_edges = np.linspace(0, len(mean_power), 9, dtype=int)
    for index, (left, right) in enumerate(zip(band_edges[:-1], band_edges[1:])):
        result[f"log_band_{index}"] = float(np.log10(mean_power[left:max(left + 1, right)].sum() / total + 1e-12))
    return result


def _select(frame: pd.DataFrame, component: str, procedural: bool, maximum: int,
            seed: int) -> pd.DataFrame:
    present = frame[f"{component}_present"].astype(int).eq(1)
    fake = frame[f"{component}_fake"].astype(int).eq(1)
    other_absent = frame[f"{'music' if component == 'voice' else 'voice'}_present"].astype(int).eq(0)
    subset = frame[present & fake & other_absent].copy()
    subset = subset[~subset.path.astype(str).str.startswith(("MIX::", "PARTIAL::"))]
    if procedural:
        subset = subset[subset.recommended_split.eq("train")]
    subset = subset.sample(frac=1.0, random_state=seed)
    return subset.head(maximum).reset_index(drop=True)


def _evaluate_component(existing: pd.DataFrame, procedural: pd.DataFrame, component: str,
                        maximum: int, seed: int) -> tuple[dict[str, object], pd.DataFrame]:
    old = _select(existing, component, False, maximum, seed)
    new = _select(procedural, component, True, maximum, seed)
    records = []
    for domain, frame in (("existing_fake", old), ("procedural_fake", new)):
        for row in frame.to_dict("records"):
            features = acoustic_features(str(row["path"]))
            records.append({"component": component, "domain": domain, "path": row["path"],
                            "group": row.get("split_group_id") or row.get("near_duplicate_group") or row["path"],
                            **features})
    feature_frame = pd.DataFrame(records)
    feature_columns = [column for column in feature_frame.columns
                       if column not in ("component", "domain", "path", "group")]
    x = feature_frame[feature_columns].to_numpy(float)
    y = feature_frame.domain.eq("procedural_fake").astype(int).to_numpy()
    groups = feature_frame.group.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(feature_frame), dtype=float)
    predictions = np.zeros(len(feature_frame), dtype=int)
    folds = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups)):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000,
                                                                          class_weight="balanced"))
        classifier.fit(x[train_index], y[train_index])
        probabilities[test_index] = classifier.predict_proba(x[test_index])[:, 1]
        predictions[test_index] = classifier.predict(x[test_index])
        folds.append({"fold": fold,
                      "auc": float(roc_auc_score(y[test_index], probabilities[test_index])),
                      "accuracy": float(accuracy_score(y[test_index], predictions[test_index]))})
    ks = {}
    for column in feature_columns:
        left = feature_frame.loc[feature_frame.domain.eq("existing_fake"), column]
        right = feature_frame.loc[feature_frame.domain.eq("procedural_fake"), column]
        statistic, pvalue = ks_2samp(left, right)
        ks[column] = {"statistic": float(statistic), "pvalue": float(pvalue),
                      "existing_mean": float(left.mean()), "procedural_mean": float(right.mean())}
    overall_auc = float(roc_auc_score(y, probabilities))
    risk = "HIGH" if overall_auc >= 0.90 else "MODERATE" if overall_auc >= 0.75 else "LOW"
    report = {
        "component": component, "existing_rows": int(len(old)), "procedural_rows": int(len(new)),
        "audio_only_source_classifier_auc": overall_auc,
        "audio_only_source_classifier_accuracy": float(accuracy_score(y, predictions)),
        "risk": risk, "folds": folds,
        "largest_distribution_shifts": dict(sorted(ks.items(), key=lambda item: item[1]["statistic"], reverse=True)[:10]),
        "interpretation": "High AUC means simple acoustic statistics expose procedural source identity.",
    }
    feature_frame["procedural_probability"] = probabilities
    return report, feature_frame


def _balance_report(base: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    labels = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
    result: dict[str, object] = {
        "base_rows": int(len(base)), "candidate_rows": int(len(candidate)),
        "base_positive_counts": {label: int(base[label].astype(int).sum()) for label in labels},
        "candidate_positive_counts": {label: int(candidate[label].astype(int).sum()) for label in labels},
    }
    procedural_mask = candidate.get("dataset_name", pd.Series("", index=candidate.index)).astype(str).eq("procedural_v8")
    for task in ("voice", "music"):
        weights = specialist_sample_weights(candidate, task)
        result[f"{task}_sampler_procedural_probability"] = float(weights[procedural_mask.to_numpy()].sum())
    result["procedural_row_fraction"] = float(procedural_mask.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-domain", type=int, default=240)
    parser.add_argument("--seed", type=int, default=23674908)
    parser.add_argument("--output", default="experiments/v8/domain_risk_report.json")
    parser.add_argument("--features", default="experiments/v8/acoustic_domain_features.csv")
    args = parser.parse_args()
    started = time.perf_counter()
    base = pd.read_csv(ROOT / "data/splits/train.csv", keep_default_na=False)
    generated = pd.read_csv(ROOT / "data/generated_v8/manifest.csv", keep_default_na=False)
    candidate = pd.read_csv(ROOT / "data/splits_v8_candidate/train.csv", keep_default_na=False)
    reports = []
    features = []
    for component in ("voice", "music"):
        report, frame = _evaluate_component(base, generated, component, args.max_per_domain, args.seed)
        reports.append(report)
        features.append(frame)
    payload = {
        "status": "MEASURED_AUDIO_ONLY_DIAGNOSTIC",
        "final_holdout": "NOT RUN",
        "feature_scope": "decoded waveform statistics only; filenames/extensions excluded",
        "components": reports,
        "balance": _balance_report(base, candidate),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.concat(features, ignore_index=True).to_csv(ROOT / args.features, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
