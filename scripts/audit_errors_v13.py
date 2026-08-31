#!/usr/bin/env python3
"""Domain-level pre-V13 error audit using only permitted non-final caches."""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ensemble import assert_final_holdout_forbidden, predict_head_selective_ensemble
from src.metrics import compute_auc, compute_eer

SPLITS = ("val_a", "val_b", "val_c", "val_d", "expanded_unseen")
BASE_SPLITS = SPLITS[:4]
OUTPUT = ROOT / "experiments/v13"


def key(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def eer_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return float(thresholds[index])


def family(generator: object) -> str:
    value = str(generator)
    if value.startswith("mix::"):
        return value.split("::")[-1]
    if "::" in value:
        return value.split("::")[-1]
    return value


def voice_kind(row: pd.Series) -> str:
    if not int(row.y_voice_present):
        return "absent"
    if not int(row.y_voice_fake):
        return "real_speech"
    value = f"{row.generator} {row.source}".lower()
    vc_tokens = ("rvc", "openvoice", "voice conversion", "freevc", "vc_")
    kind = "VC" if any(token in value for token in vc_tokens) else "TTS_or_unknown"
    if "unseen" in str(row.data_role).lower() or row.split == "expanded_unseen":
        return f"unseen_{kind}"
    return kind


def language(path: object) -> str:
    value = "/" + key(path).strip("/") + "/"
    for code in ("ko", "en", "de", "fr", "es", "it", "pl", "ru", "uk"):
        if f"/{code}/" in value:
            return code
    return "unknown"


def music_kind(row: pd.Series) -> str:
    if not int(row.y_music_present):
        return "absent"
    if not int(row.y_music_fake):
        return "real_music"
    value = family(row.generator).lower()
    for token, name in (
        ("musicgen", "MusicGen"), ("sonics", "SONICS"), ("chirp", "SONICS"),
        ("suno", "Suno-like"), ("udio", "Udio-like"),
        ("audioldm", "AudioLDM"), ("stable", "StableAudio-like"),
        ("acestep", "ACE-Step"), ("echoes", "Echoes-other"),
    ):
        if token in value:
            return name
    return "other_fake_music"


def load_metadata() -> dict[str, dict]:
    paths = [ROOT / f"data/splits_v12/{name}.csv" for name in BASE_SPLITS]
    paths.append(ROOT / "data/splits_v9_candidate/manifest.csv")
    metadata: dict[str, dict] = {}
    for path in paths:
        frame = pd.read_csv(path)
        for record in frame.to_dict("records"):
            metadata.setdefault(key(record.get("path", "")), record)
    return metadata


def enrich(frame: pd.DataFrame, metadata: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for record in frame.to_dict("records"):
        extra = metadata.get(key(record["path"]), {})
        combined = dict(extra)
        combined.update(record)
        rows.append(combined)
    result = pd.DataFrame(rows)
    for column, default in (
        ("mix_mode", "none"), ("augment", "none"),
        ("mix_overlap_fraction", np.nan), ("upstream_label", "unknown"),
    ):
        if column not in result:
            result[column] = default
        result[column] = result[column].fillna(default)
    return result


def metric_summary(group: pd.DataFrame, label: str, score: str,
                   threshold: float, present: str | None = None) -> dict:
    if present is not None:
        group = group[group[present].astype(int).eq(1)]
    labels = group[label].to_numpy(dtype=int)
    scores = group[score].to_numpy(dtype=float)
    result = {
        "samples": int(len(group)),
        "negative": int(np.sum(labels == 0)),
        "positive": int(np.sum(labels == 1)),
        "eer": float(compute_eer(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "auc": float(compute_auc(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "false_positive": int(np.sum((labels == 0) & (scores >= threshold))),
        "false_negative": int(np.sum((labels == 1) & (scores < threshold))),
        "mean_score": float(np.mean(scores)) if len(scores) else None,
    }
    return result


def domain_summary(frame: pd.DataFrame, dimension: str, thresholds: dict) -> dict:
    output = {}
    for value, group in frame.groupby(dimension, dropna=False, sort=True):
        output[str(value)] = {
            "file": metric_summary(group, "y_file_fake", "p_file", thresholds["file"]),
            "voice": metric_summary(
                group, "y_voice_fake", "p_voice", thresholds["voice"], "y_voice_present"),
            "music": metric_summary(
                group, "y_music_fake", "p_music", thresholds["music"], "y_music_present"),
        }
    return output


def main() -> None:
    assert_final_holdout_forbidden(OUTPUT)
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text(encoding="utf-8"))
    metadata = load_metadata()
    frames = []
    for split in SPLITS:
        cache = ROOT / f"experiments/v12/cache/v7_canonical/{split}.csv"
        assert_final_holdout_forbidden(cache)
        frame = enrich(pd.read_csv(cache), metadata)
        prediction = predict_head_selective_ensemble(frame, weights)
        frame[["p_file", "p_voice", "p_music", "p_voice_present", "p_music_present"]] = prediction
        frame["split"] = split
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    base = samples[samples["split"].isin(BASE_SPLITS)].copy()
    thresholds = {
        "file": eer_threshold(base.y_file_fake.to_numpy(int), base.p_file.to_numpy(float)),
        "voice": eer_threshold(
            base.loc[base.y_voice_present.eq(1), "y_voice_fake"].to_numpy(int),
            base.loc[base.y_voice_present.eq(1), "p_voice"].to_numpy(float)),
        "music": eer_threshold(
            base.loc[base.y_music_present.eq(1), "y_music_fake"].to_numpy(int),
            base.loc[base.y_music_present.eq(1), "p_music"].to_numpy(float)),
    }
    samples["voice_domain"] = samples.apply(voice_kind, axis=1)
    samples["voice_language"] = samples.path.map(language)
    samples["voice_channel"] = np.select(
        [samples["augment"].astype(str).str.contains("telephone", case=False),
         samples["augment"].astype(str).str.contains("codec", case=False)],
        ["telephone", "codec"], default="clean_or_unknown")
    samples["voice_mixed_music"] = np.where(samples.y_music_present.eq(1), "with_music", "voice_only")
    samples["music_domain"] = samples.apply(music_kind, axis=1)
    samples["music_generator_family"] = samples.generator.map(family)
    samples["file_composition"] = np.select(
        [samples.y_voice_present.eq(1) & samples.y_music_present.eq(1),
         samples.y_voice_present.eq(1), samples.y_music_present.eq(1)],
        ["voice+music", "voice_only", "music_only"], default="neither")
    samples["file_state"] = np.where(
        samples.file_composition.eq("voice+music"),
        np.select(
            [samples.y_voice_fake.eq(0) & samples.y_music_fake.eq(0),
             samples.y_voice_fake.eq(0) & samples.y_music_fake.eq(1),
             samples.y_voice_fake.eq(1) & samples.y_music_fake.eq(0)],
            ["RR", "RF", "FR"], default="FF"),
        np.where(samples.y_file_fake.eq(1), "single_fake", "single_real"))
    samples["file_mix_mode"] = samples.mix_mode.astype(str).replace({"nan": "none", "": "none"})
    samples["file_partial"] = np.where(
        samples.file_mix_mode.str.contains("partial|crossfade", case=False),
        "partial_or_crossfade", "not_partial_or_unknown")
    samples["duration_bucket"] = pd.cut(
        samples.duration_sec, [-np.inf, 2, 5, 10, 20, np.inf],
        labels=("<=2", "2-5", "5-10", "10-20", ">20")).astype(str)
    samples["generator_family"] = samples.generator.map(family)
    for head, label, score in (
        ("file", "y_file_fake", "p_file"),
        ("voice", "y_voice_fake", "p_voice"),
        ("music", "y_music_fake", "p_music"),
    ):
        eligible = np.ones(len(samples), dtype=bool)
        if head != "file":
            eligible = samples[f"y_{head}_present"].to_numpy(int) == 1
        samples[f"{head}_error"] = np.where(
            eligible,
            np.where((samples[label].to_numpy(int) == 0) &
                     (samples[score].to_numpy(float) >= thresholds[head]), "FP",
              np.where((samples[label].to_numpy(int) == 1) &
                       (samples[score].to_numpy(float) < thresholds[head]), "FN", "correct")),
            "not_scored")

    dimensions = (
        "split", "source", "generator_family", "voice_domain", "voice_language",
        "voice_channel", "voice_mixed_music", "music_domain", "music_generator_family",
        "file_composition", "file_state", "file_mix_mode", "file_partial", "duration_bucket",
    )
    report = {
        "status": "MEASURED_NON_FINAL_PRE_V13_ERROR_AUDIT",
        "final_holdout": "NOT READ / NOT RUN",
        "selected_pipeline": "TEST5/V7",
        "thresholds_fit_on": list(BASE_SPLITS),
        "eer_thresholds": thresholds,
        "overall_val": {
            "file": metric_summary(base, "y_file_fake", "p_file", thresholds["file"]),
            "voice": metric_summary(base, "y_voice_fake", "p_voice", thresholds["voice"], "y_voice_present"),
            "music": metric_summary(base, "y_music_fake", "p_music", thresholds["music"], "y_music_present"),
        },
        "domains": {dimension: domain_summary(samples, dimension, thresholds)
                    for dimension in dimensions},
        "coverage_limits": {
            "korean": "only reported when language is encoded in the public path; otherwise unknown",
            "noise_snr": "NOT AVAILABLE in canonical cache metadata",
            "music_vocality": "NOT AVAILABLE in canonical cache metadata",
            "fake_occupancy": "NOT AVAILABLE for most legacy rows",
            "partial_fake": "mix-mode proxy only; dedicated V13 partial set required",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUTPUT / "pre_v13_error_samples.csv", index=False)
    (OUTPUT / "pre_v13_error_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(samples), "thresholds": thresholds,
                      "overall_val": report["overall_val"]}, indent=2))


if __name__ == "__main__":
    main()
