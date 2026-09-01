#!/usr/bin/env python3
"""Leakage audit for the direct, paired V13B production core.

Virtual PARTIAL/MIX rows are audited structurally elsewhere because their path
syntax and combinatorial labels are not file-source metadata.  This audit uses
content-grouped folds so counterparts never cross train/test within the audit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
from tools.v13_guards import assert_final_holdout_v13b_forbidden

METADATA_NUMERIC = ("sample_rate", "duration", "bytes_per_second")
ACOUSTIC = ("rms", "silence_ratio", "clipping_ratio", "zcr", "spectral_centroid",
            "spectral_rolloff", "spectral_flatness", "high_frequency_ratio")
CATEGORICAL = ("source", "dataset", "extension", "codec", "augment", "channel_policy")


def direct_path(value: object) -> pathlib.Path:
    path = pathlib.Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract(row: pd.Series) -> dict:
    path = direct_path(row.path)
    info = sf.info(path)
    wave = np.asarray(load_manifest_row_wave(
        row, sr=16_000, is_training=False, use_demucs=False), dtype=np.float32)
    # All examples get exactly the same four-second observation policy.
    if len(wave) > 64_000:
        start = (len(wave) - 64_000) // 2
        wave = wave[start:start + 64_000]
    elif len(wave) < 64_000:
        wave = np.pad(wave, (0, 64_000 - len(wave)))
    windowed = wave * np.hanning(len(wave))
    magnitude = np.abs(np.fft.rfft(windowed)) + 1e-10
    power = magnitude ** 2
    frequencies = np.fft.rfftfreq(len(wave), 1.0 / 16_000)
    cumulative = np.cumsum(power)
    rolloff = frequencies[min(
        np.searchsorted(cumulative, cumulative[-1] * 0.85), len(frequencies) - 1)]
    high = power[frequencies >= 4_000].sum()
    total = power.sum()
    return {
        "path": str(path), "label": int(row.file_fake),
        "content_group": str(row.content_group), "source": str(row.source),
        "dataset": str(row.get("dataset", row.source)),
        "extension": path.suffix.lower(), "codec": str(row.get("codec", info.subtype)),
        "augment": str(row.get("augment", "none")),
        "channel_policy": str(row.get("channel_policy", "unknown")),
        "sample_rate": float(info.samplerate), "duration": float(info.duration),
        "bytes_per_second": float(path.stat().st_size / max(info.duration, 1e-6)),
        "rms": float(np.sqrt(np.mean(wave ** 2))),
        "silence_ratio": float(np.mean(np.abs(wave) < 1e-3)),
        "clipping_ratio": float(np.mean(np.abs(wave) >= 0.999)),
        "zcr": float(np.mean(np.signbit(wave[1:]) != np.signbit(wave[:-1]))),
        "spectral_centroid": float(np.sum(frequencies * magnitude) / magnitude.sum()),
        "spectral_rolloff": float(rolloff),
        "spectral_flatness": float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude)),
        "high_frequency_ratio": float(high / (total + 1e-12)),
    }


def pipeline(numeric: tuple[str, ...], categorical: tuple[str, ...]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), list(numeric)))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(
            handle_unknown="ignore", min_frequency=2), list(categorical)))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced")),
    ])


def grouped_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=23674913)
    return list(splitter.split(frame, frame.label, groups=frame.content_group))


def auc(frame: pd.DataFrame, numeric: tuple[str, ...], categorical: tuple[str, ...],
        folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    probability = cross_val_predict(
        pipeline(numeric, categorical), frame, frame.label.to_numpy(int), cv=folds,
        method="predict_proba")[:, 1]
    return float(roc_auc_score(frame.label, probability))


def ablations(frame: pd.DataFrame, numeric: tuple[str, ...], categorical: tuple[str, ...],
              folds: list[tuple[np.ndarray, np.ndarray]], baseline: float) -> list[dict]:
    result = []
    for feature in (*numeric, *categorical):
        value = auc(frame, tuple(item for item in numeric if item != feature),
                    tuple(item for item in categorical if item != feature), folds)
        result.append({"feature": feature, "auc_without": value,
                       "auc_drop": float(baseline - value)})
    return sorted(result, key=lambda item: item["auc_drop"], reverse=True)


def source_prediction(frame: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    model = pipeline(ACOUSTIC, ())
    predicted = cross_val_predict(model, frame, frame.source.astype(str), cv=folds, method="predict")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(frame.source.astype(str), predicted)),
        "classes": sorted(frame.source.astype(str).unique()),
        "chance_reference": float(1.0 / frame.source.nunique()),
        "features": list(ACOUSTIC),
    }


def suggestions(top: list[dict]) -> list[str]:
    names = {item["feature"] for item in top[:5] if item["auc_drop"] > 0.002}
    output = []
    if names & {"source", "dataset"}:
        output.append("increase same-source real/fake pairs and rebalance each source")
    if names & {"sample_rate", "codec", "extension", "bytes_per_second"}:
        output.append("canonicalize container/sample-rate/codec identically for both labels")
    if "duration" in names:
        output.append("use identical crop/pad policy and duration-stratified sampling")
    if names & set(ACOUSTIC):
        output.append("apply label-independent gain/channel augmentation and inspect source acoustic fingerprints")
    if not output:
        output.append("no single metadata feature dominates; add another paired source before scaling")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits_v13b/train.csv")
    parser.add_argument("--output", default="experiments/v13b/source_shortcut_audit.json")
    args = parser.parse_args()
    manifest = ROOT / args.manifest
    output = ROOT / args.output
    assert_final_holdout_v13b_forbidden(manifest, output)
    raw = pd.read_csv(manifest)
    direct = raw[raw.data_role.eq("paired_core") &
                 ~raw.path.astype(str).str.startswith(("MIX::", "PARTIAL::"))].copy()
    extracted = pd.DataFrame([extract(row) for _, row in direct.iterrows()])
    if extracted.label.nunique() != 2 or extracted.source.nunique() < 2:
        raise RuntimeError("V13B shortcut audit requires both labels and at least two sources")
    folds = grouped_folds(extracted)
    metadata_numeric = METADATA_NUMERIC
    metadata_auc = auc(extracted, metadata_numeric, CATEGORICAL, folds)
    acoustic_auc = auc(extracted, ACOUSTIC, (), folds)
    combined_numeric = METADATA_NUMERIC + ACOUSTIC
    combined_auc = auc(extracted, combined_numeric, CATEGORICAL, folds)
    metadata_ablation = ablations(
        extracted, metadata_numeric, CATEGORICAL, folds, metadata_auc)
    acoustic_ablation = ablations(extracted, ACOUSTIC, (), folds, acoustic_auc)
    hard_threshold = 0.75
    passed = max(metadata_auc, acoustic_auc, combined_auc) <= hard_threshold
    features_path = output.with_name(output.stem + "_features.csv")
    fingerprint_path = output.with_name(output.stem + "_source_label_fingerprints.csv")
    fingerprints = extracted.groupby(["source", "label"], as_index=False)[
        ["duration", "zcr", "rms", "spectral_centroid", "silence_ratio",
         "high_frequency_ratio"]].agg(["count", "mean", "std"])
    fingerprints.columns = ["_".join(str(part) for part in column if part)
                            for column in fingerprints.columns.to_flat_index()]
    report = {
        "status": "PASS" if passed else "FAIL_SOURCE_SHORTCUT",
        "decision": "DATASET_GATE_PASS" if passed else "MODEL_TRAINING_BLOCKED",
        "scope": "TRAIN direct paired_core only; virtual partial/mixed rows structurally audited separately",
        "manifest": args.manifest, "rows": len(extracted), "folds": 5,
        "fold_isolation": "StratifiedGroupKFold(content_group)",
        "hard_threshold": hard_threshold,
        "threshold_policy": {"hard_combined_max": 0.75, "target_combined_max": 0.65,
                             "ideal_combined_range": [0.50, 0.60]},
        "recommended_thresholds": {"metadata": 0.60, "acoustic": 0.65,
                                   "combined": 0.65},
        "metadata_only": {"auc": metadata_auc, "numeric": list(metadata_numeric),
                          "categorical": list(CATEGORICAL)},
        "acoustic_only": {"auc": acoustic_auc, "features": list(ACOUSTIC)},
        "combined": {"auc": combined_auc},
        "source_prediction_acoustic": source_prediction(extracted, folds),
        "metadata_feature_ablation": metadata_ablation,
        "acoustic_feature_ablation": acoustic_ablation,
        "likely_causes": [item for item in (metadata_ablation + acoustic_ablation)
                          if item["auc_drop"] > 0.002][:8],
        "automatic_remediation_candidates": suggestions(metadata_ablation + acoustic_ablation),
        "class_balance": extracted.label.value_counts().sort_index().to_dict(),
        "source_label_balance": extracted.groupby(["source", "label"]).size().unstack(fill_value=0).to_dict(),
        "features_file": features_path.relative_to(ROOT).as_posix(),
        "source_label_fingerprints_file": fingerprint_path.relative_to(ROOT).as_posix(),
        "final_holdout_v13b": "NOT READ / NOT RUN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(features_path, index=False)
    fingerprints.to_csv(fingerprint_path, index=False)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    dataset_path = ROOT / "data/splits_v13b/DATASET_V13B.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["shortcut_gate"] = {
        "status": report["status"], "metadata_auc": metadata_auc,
        "acoustic_auc": acoustic_auc, "combined_auc": combined_auc,
    }
    dataset["structural_gates"]["metadata_auc_at_most_0_75"] = metadata_auc <= hard_threshold
    dataset["structural_gates"]["acoustic_auc_at_most_0_75"] = acoustic_auc <= hard_threshold
    dataset["structural_gates"]["combined_auc_at_most_0_75"] = combined_auc <= hard_threshold
    dataset["known_blockers"] = [key for key, value in dataset["structural_gates"].items()
                                 if not value]
    dataset["status"] = "DATASET_READY" if not dataset["known_blockers"] else "DATASET_NOT_READY"
    dataset["model_training"] = ("ALLOWED_BY_DATA_GATES" if dataset["status"] == "DATASET_READY"
                                 else "BLOCKED_BY_DATA_GATES")
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
