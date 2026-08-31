#!/usr/bin/env python3
"""Domain-level VOICE FP/FN analysis using non-final VAL-A/B/C/D caches only."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import compute_eer

SPLITS = ("val_a", "val_b", "val_c", "val_d")


def eer_operating_point(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return float(thresholds[index]), float((fpr[index] + 1.0 - tpr[index]) / 2.0)


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def summarize(group):
    labels = group["voice_fake"].to_numpy(dtype=int)
    scores = group["voice_score"].to_numpy(dtype=float)
    counts = np.bincount(labels, minlength=2)
    return {
        "samples": int(len(group)), "real": int(counts[0]), "fake": int(counts[1]),
        "eer": float(compute_eer(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "false_positive": int(group["false_positive"].sum()),
        "false_negative": int(group["false_negative"].sum()),
        "mean_score": finite_or_none(np.mean(scores)),
        "score_std": finite_or_none(np.std(scores)),
        "score_p10": finite_or_none(np.quantile(scores, 0.10)),
        "score_p50": finite_or_none(np.quantile(scores, 0.50)),
        "score_p90": finite_or_none(np.quantile(scores, 0.90)),
    }


def main():
    output_dir = ROOT / "experiments/v7"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    component_weight = float(weights.get("w_df_voice_component", weights.get("w_df_component", 0.0)))
    frames = []
    split_summary = {}
    for split in SPLITS:
        feature = pd.read_csv(ROOT / "experiments" / f"{split}_features_16k.csv")
        aggregation = pd.read_csv(ROOT / "experiments" / f"{split}_voice_aggregation.csv")
        v7_feature_path = ROOT / "experiments/v7" / f"{split}_voice_features.csv"
        if v7_feature_path.exists():
            v7_voice = pd.read_csv(v7_feature_path)
            if not v7_voice["path"].astype(str).equals(aggregation["path"].astype(str)):
                raise RuntimeError(f"{split}: v7 voice-feature ordering mismatch")
            aggregation = aggregation.copy()
            aggregation["vf_max"] = v7_voice["vf_max"].to_numpy(float)
        if len(feature) != len(aggregation):
            raise RuntimeError(f"{split}: feature/aggregation row count mismatch")
        source = pd.read_csv(ROOT / "data/splits" / f"{split}.csv")
        if source["path"].astype(str).duplicated().any():
            raise RuntimeError(f"{split}: duplicate path prevents exact analysis join")
        source = source.set_index(source["path"].astype(str), drop=False)
        paths = aggregation["path"].astype(str)
        missing = set(paths) - set(source.index)
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} cached paths missing from split")
        meta = source.loc[paths].reset_index(drop=True)
        for column in ("voice_fake", "voice_present"):
            cached = feature[f"y_{column}"].to_numpy(dtype=int)
            if not np.array_equal(cached, meta[column].to_numpy(dtype=int)):
                raise RuntimeError(f"{split}: cache label mismatch for {column}")
        present = meta["voice_present"].to_numpy(dtype=int) == 1
        frame = pd.DataFrame({
            "split": split,
            "path": paths,
            "dataset": meta.get("dataset", pd.Series("unknown", index=meta.index)).astype(str),
            "source": meta.get("source", pd.Series("unknown", index=meta.index)).astype(str),
            "generator": meta.get("generator", pd.Series("unknown", index=meta.index)).astype(str),
            "speaker_id": meta.get("speaker_id", pd.Series("unknown", index=meta.index)).astype(str),
            "voice_fake": meta["voice_fake"].astype(int),
            "voice_present": meta["voice_present"].astype(int),
            "music_present": meta["music_present"].astype(int),
            "duration_sec": feature["duration_sec"].astype(float),
            "specialist_score": aggregation["vf_max"].astype(float),
            "df_score": feature["df_primary"].astype(float),
            "voice_score": component_weight * feature["df_primary"].astype(float)
                           + (1.0 - component_weight) * aggregation["vf_max"].astype(float),
            "segment_spread": aggregation["vf_max"].astype(float) - aggregation["vf_mean"].astype(float),
            "mix_mode": meta.get("mix_mode", pd.Series("none", index=meta.index)).fillna("none").astype(str),
            "mix_snr_db": pd.to_numeric(meta.get("mix_snr_db", np.nan), errors="coerce"),
            "augment": meta.get("augment", pd.Series("none", index=meta.index)).fillna("none").astype(str),
        })
        frame = frame[present].reset_index(drop=True)
        threshold, eer = eer_operating_point(frame["voice_fake"], frame["voice_score"])
        frame["threshold"] = threshold
        frame["predicted_fake"] = (frame["voice_score"] >= threshold).astype(int)
        frame["false_positive"] = ((frame["voice_fake"] == 0) & (frame["predicted_fake"] == 1)).astype(int)
        frame["false_negative"] = ((frame["voice_fake"] == 1) & (frame["predicted_fake"] == 0)).astype(int)
        frame["error_type"] = np.select(
            [frame["false_positive"].eq(1), frame["false_negative"].eq(1)], ["FP", "FN"], default="correct")
        frame["duration_bucket"] = pd.cut(
            frame["duration_sec"], [-np.inf, 4, 8, 15, 30, np.inf],
            labels=("<=4", "4-8", "8-15", "15-30", ">30")).astype(str)
        frame["snr_bucket"] = pd.cut(
            frame["mix_snr_db"], [-np.inf, -5, 5, np.inf], labels=("low", "mid", "high")).astype(str)
        frame.loc[frame["mix_snr_db"].isna(), "snr_bucket"] = "not_mixed"
        frame["codec"] = frame["augment"].str.contains("codec", case=False)
        frame["telephone"] = frame["augment"].str.contains("telephone|tel", case=False)
        frame["mixed_music"] = frame["music_present"].eq(1)
        frame["partial_overlap"] = frame["mix_mode"].eq("partial_overlap")
        frame["confidence_bucket"] = pd.cut(
            np.abs(frame["voice_score"] - threshold), [-np.inf, .05, .15, np.inf],
            labels=("uncertain", "moderate", "confident")).astype(str)
        split_summary[split] = {"threshold": threshold, **summarize(frame), "eer_at_threshold": eer}
        frames.append(frame)

    samples = pd.concat(frames, ignore_index=True)
    dimensions = (
        "split", "voice_fake", "generator", "speaker_id", "dataset", "source",
        "duration_bucket", "snr_bucket", "codec", "telephone", "mixed_music",
        "partial_overlap", "mix_mode", "confidence_bucket",
    )
    grouped = {}
    for dimension in dimensions:
        grouped[dimension] = {
            str(key): summarize(group) for key, group in samples.groupby(dimension, dropna=False)
        }
    report = {
        "status": "MEASURED_NON_FINAL_LOCAL_VALIDATION",
        "final_holdout": "NOT RUN",
        "source_caches": [f"experiments/{split}_features_16k.csv" for split in SPLITS],
        "voice_score": f"{component_weight:.6g} * DF + {1.0-component_weight:.6g} * specialist_max",
        "splits": split_summary,
        "groups": grouped,
        "overall": summarize(samples),
    }
    samples.sort_values(["error_type", "split", "voice_score"], ascending=[True, True, False]).to_csv(
        output_dir / "voice_error_samples.csv", index=False)
    (output_dir / "voice_error_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"splits": split_summary, "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
