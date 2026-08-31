#!/usr/bin/env python3
"""Emit immutable v7 provenance/diversity facts without approving unknown data."""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def count_table(frame, keys):
    return [dict(zip(keys, values if isinstance(values, tuple) else (values,)), samples=int(count))
            for values, count in frame.groupby(keys, dropna=False).size().items()]


def main():
    output = ROOT / "experiments/v7"
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "data/manifest.csv")
    status = manifest["allowed_for_competition"].astype(str).map(
        lambda value: "APPROVED" if value == "YES" else (
            "REJECTED" if value in ("NO", "REJECTED") else "REVIEW_REQUIRED"))
    manifest = manifest.assign(provenance_status=status)
    by_dataset = count_table(manifest, ["dataset_name", "provenance_status"])
    voice = manifest[manifest["voice_present"].eq(1)]
    diversity = []
    for keys, group in voice.groupby(["dataset_name", "generator", "voice_fake"], dropna=False):
        diversity.append({
            "dataset_name": str(keys[0]), "generator": str(keys[1]), "voice_fake": int(keys[2]),
            "samples": int(len(group)), "unique_originals": int(group["original_id"].nunique()),
            "unique_speakers": int(group["speaker_id"].nunique()),
            "unique_split_groups": int(group["split_group_id"].nunique()),
        })
    approved = manifest[manifest["provenance_status"].eq("APPROVED")]
    approved_voice_labels = sorted(approved.loc[approved["voice_present"].eq(1), "voice_fake"].unique().tolist())
    payload = {
        "status": "MEASURED_MANIFEST_AUDIT",
        "rows": int(len(manifest)),
        "provenance_status": {str(k): int(v) for k, v in status.value_counts().items()},
        "by_dataset": by_dataset,
        "voice_diversity": diversity,
        "approved_only_training": {
            "status": "NOT RUN",
            "reason": "approved subset has no fake voice/music class and cannot form metric-complete leakage-safe splits",
            "rows": int(len(approved)), "voice_fake_labels": approved_voice_labels,
        },
        "policy": "REVIEW_REQUIRED is never promoted automatically; human rule/license review is required",
    }
    (output / "provenance_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "data_diversity.json").write_text(json.dumps({
        "status": payload["status"], "voice_diversity": diversity,
        "final_holdout": "NOT USED",
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
