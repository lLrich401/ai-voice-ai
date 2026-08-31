#!/usr/bin/env python3
"""Score the generator-disjoint procedural positive stress set with selected v7."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import script as submission  # noqa: E402


OUTPUT_COLUMNS = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                  "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")


def _positive_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)), "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "recall_at_0_5": float(np.mean(values >= 0.5)),
        "false_negative_rate_at_0_5": float(np.mean(values < 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/generated_v8/manifest.csv")
    parser.add_argument("--batch-files", type=int, default=16)
    parser.add_argument("--output", default="experiments/v8/v7_procedural_stress_report.json")
    parser.add_argument("--predictions", default="experiments/v8/v7_procedural_stress_predictions.csv")
    args = parser.parse_args()
    frame = pd.read_csv(ROOT / args.manifest, keep_default_na=False)
    frame = frame[frame.recommended_split.eq("val_unseen_generator")].copy()
    paths = [ROOT / path for path in frame.path.astype(str)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    submission.verify_mandatory_models()
    voice_model = submission.load_voice_model(device)
    music_model = submission.load_music_model(device)
    df_session = submission.load_df_arena(device)
    panns_model = submission.load_panns(device)
    weights = submission.load_fusion_weights()
    started = time.perf_counter()
    rows = []
    for lower in range(0, len(paths), args.batch_files):
        rows.extend(submission.infer_files_batch(
            voice_model, music_model, df_session, panns_model, weights,
            paths[lower:lower + args.batch_files], device, use_demucs=False))
    elapsed = time.perf_counter() - started
    predictions = pd.DataFrame(rows, columns=("id", *OUTPUT_COLUMNS))
    label_by_id = frame.assign(id=frame.path.map(lambda value: pathlib.Path(value).stem)).set_index("id")
    if predictions.id.duplicated().any() or set(predictions.id) != set(label_by_id.index):
        raise RuntimeError("prediction/manifest ID mismatch")
    merged = label_by_id.loc[predictions.id].reset_index().merge(predictions, on="id", validate="one_to_one")
    component = {}
    target_columns = {
        "file_fake": ("file_fake", "FILE_FAKE_PROB", np.ones(len(merged), dtype=bool)),
        "voice_fake": ("voice_fake", "VOICE_FAKE_PROB", merged.voice_present.astype(int).eq(1).to_numpy()),
        "music_fake": ("music_fake", "MUSIC_FAKE_PROB", merged.music_present.astype(int).eq(1).to_numpy()),
    }
    for name, (_, prediction, mask) in target_columns.items():
        component[name] = _positive_summary(merged.loc[mask, prediction].to_numpy(float))
    presence = {}
    for component_name, target, prediction in (
        ("voice", "voice_present", "VOICE_PRESENT_PROB"),
        ("music", "music_present", "MUSIC_PRESENT_PROB"),
    ):
        y = merged[target].astype(int).to_numpy()
        score = merged[prediction].to_numpy(float)
        presence[component_name] = {
            "auc": float(roc_auc_score(y, score)),
            "accuracy_at_0_5": float(np.mean((score >= 0.5) == y)),
            "present_recall_at_0_5": float(np.mean(score[y == 1] >= 0.5)),
            "absent_specificity_at_0_5": float(np.mean(score[y == 0] < 0.5)),
        }
    by_kind = {}
    for kind, group in merged.groupby(["voice_present", "music_present"]):
        label = {(1, 0): "voice_only", (0, 1): "music_only", (1, 1): "mixed"}.get(tuple(map(int, kind)), str(kind))
        by_kind[label] = {column: float(group[column].mean()) for column in OUTPUT_COLUMNS}
    report = {
        "status": "MEASURED_GENERATOR_DISJOINT_POSITIVE_STRESS",
        "selected_pipeline": weights.get("pipeline_version"), "final_holdout": "NOT RUN",
        "rows": int(len(merged)), "runtime_seconds": float(elapsed),
        "seconds_per_file": float(elapsed / len(merged)),
        "fake_detection": component, "presence": presence, "mean_predictions_by_kind": by_kind,
        "interpretation": "Positive-only stress recall is not EER and cannot replace VAL-A/B/C/D.",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    merged.to_csv(ROOT / args.predictions, index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
