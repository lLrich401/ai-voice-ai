#!/usr/bin/env python3
"""Detect label leakage from source/metadata and shallow acoustic fingerprints."""

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
from src.ensemble import assert_final_holdout_forbidden

NUMERIC = (
    "sample_rate", "duration", "bytes_per_second", "rms", "silence_ratio",
    "clipping_ratio", "zcr", "spectral_centroid", "spectral_rolloff",
    "spectral_flatness", "high_frequency_ratio",
)
CATEGORICAL = ("source", "dataset", "extension", "augment", "is_mix")


def direct_path(value: object) -> pathlib.Path | None:
    text = str(value)
    if text.startswith("MIX::"):
        return None
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.is_file() else None


def features(row: pd.Series) -> dict:
    path = direct_path(row.path)
    sample_rate, duration, bytes_per_second = np.nan, np.nan, np.nan
    if path is not None:
        try:
            info = sf.info(path)
            sample_rate = float(info.samplerate)
            duration = float(info.duration)
            bytes_per_second = float(path.stat().st_size / max(duration, 1e-6))
        except Exception:
            pass
    wave = np.asarray(load_manifest_row_wave(
        row, sr=16000, is_training=False, use_demucs=False), dtype=np.float32)
    if len(wave) > 64000:
        start = (len(wave) - 64000) // 2
        wave = wave[start:start + 64000]
    if len(wave) < 64000:
        wave = np.pad(wave, (0, 64000 - len(wave)))
    magnitude = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) + 1e-10
    power = magnitude ** 2
    frequencies = np.fft.rfftfreq(len(wave), 1.0 / 16000)
    cumulative = np.cumsum(power)
    rolloff = frequencies[min(np.searchsorted(cumulative, cumulative[-1] * 0.85), len(frequencies) - 1)]
    low = power[frequencies < 4000].sum()
    high = power[frequencies >= 4000].sum()
    return {
        "path": str(row.path),
        "label": int(row.file_fake),
        "source": str(row.get("source", "unknown")),
        "dataset": str(row.get("dataset", "unknown")),
        "extension": pathlib.Path(str(row.path).split("|")[0]).suffix.lower() or "mixed",
        "augment": str(row.get("augment", "none")),
        "is_mix": str(str(row.path).startswith("MIX::")),
        "sample_rate": sample_rate,
        "duration": duration if np.isfinite(duration) else float(len(wave) / 16000),
        "bytes_per_second": bytes_per_second,
        "rms": float(np.sqrt(np.mean(wave ** 2))),
        "silence_ratio": float(np.mean(np.abs(wave) < 1e-3)),
        "clipping_ratio": float(np.mean(np.abs(wave) >= 0.999)),
        "zcr": float(np.mean(np.signbit(wave[1:]) != np.signbit(wave[:-1]))),
        "spectral_centroid": float(np.sum(frequencies * magnitude) / magnitude.sum()),
        "spectral_rolloff": float(rolloff),
        "spectral_flatness": float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude)),
        "high_frequency_ratio": float(high / (low + high + 1e-12)),
    }


def balanced_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame.sample(frac=1.0, random_state=20260913).reset_index(drop=True)
    chosen: list[int] = []
    groups = list(frame.groupby(["source", "file_fake"], dropna=False))
    quota = max(2, limit // max(len(groups), 1))
    for _, group in groups:
        chosen.extend(group.sample(
            min(len(group), quota), random_state=20260913).index.tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < limit:
        remaining = frame.drop(index=chosen, errors="ignore")
        chosen.extend(remaining.sample(
            min(limit - len(chosen), len(remaining)), random_state=20260914).index.tolist())
    return frame.loc[chosen[:limit]].reset_index(drop=True)


def cv_auc(frame: pd.DataFrame, numeric: tuple[str, ...],
           categorical: tuple[str, ...]) -> dict:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(numeric)))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(
            handle_unknown="ignore", min_frequency=2), list(categorical)))
    model = Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(max_iter=2000, C=0.2, class_weight="balanced")),
    ])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260913)
    probabilities = cross_val_predict(
        model, frame, frame.label.to_numpy(int), cv=folds, method="predict_proba")[:, 1]
    return {
        "auc": float(roc_auc_score(frame.label, probabilities)),
        "folds": 5,
        "classifier": "L2 logistic regression",
        "features": list(numeric + categorical),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits_v12/train.csv")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--output", default="experiments/v13/source_shortcut_audit.json")
    args = parser.parse_args()
    manifest = ROOT / args.manifest
    output = ROOT / args.output
    assert_final_holdout_forbidden(manifest, output)
    source = balanced_sample(pd.read_csv(manifest), args.limit)
    extracted = pd.DataFrame([features(row) for _, row in source.iterrows()])
    metadata = cv_auc(extracted, (), CATEGORICAL)
    acoustic = cv_auc(extracted, NUMERIC, ())
    combined = cv_auc(extracted, NUMERIC, CATEGORICAL)
    threshold = 0.75
    features_path = output.with_name(f"{output.stem}_features.csv")
    report = {
        "status": "FAIL_SOURCE_SHORTCUT" if max(metadata["auc"], combined["auc"]) > threshold else "PASS",
        "threshold": threshold,
        "manifest": args.manifest,
        "rows": int(len(extracted)),
        "features_file": features_path.relative_to(ROOT).as_posix(),
        "class_balance": extracted.label.value_counts().sort_index().to_dict(),
        "source_balance": extracted.source.value_counts().to_dict(),
        "metadata_only": metadata,
        "acoustic_only": acoustic,
        "combined": combined,
        "interpretation": (
            "AUC above threshold means label is recoverable from dataset/channel fingerprints; "
            "the dataset must not be treated as a forensic-generalization success."),
        "final_holdout": "NOT READ / NOT RUN",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(features_path, index=False)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
