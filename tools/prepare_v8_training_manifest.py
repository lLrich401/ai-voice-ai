#!/usr/bin/env python3
"""Create isolated v8 candidate manifests without modifying selected v7 splits."""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _identifiers(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    return set(frame[column].fillna("").astype(str)) - {""}


def prepare(base_train_path: pathlib.Path, generated_path: pathlib.Path,
            output_dir: pathlib.Path, risk_report_path: pathlib.Path | None = None) -> dict[str, object]:
    base = pd.read_csv(base_train_path, keep_default_na=False)
    generated = pd.read_csv(generated_path, keep_default_na=False)
    required = {"path", "recommended_split", "external_assets_used", "allowed_for_competition",
                "split_group_id", "near_duplicate_group", "generator_family"}
    missing = sorted(required - set(generated.columns))
    if missing:
        raise ValueError(f"generated manifest missing columns: {missing}")
    unsafe = generated[(generated.external_assets_used != "NO") |
                       (generated.allowed_for_competition != "YES")]
    if len(unsafe):
        raise ValueError(f"unsafe generated provenance rows: {len(unsafe)}")
    generated_train = generated[generated.recommended_split == "train"].copy()
    generated_valid = generated[generated.recommended_split == "val_unseen_generator"].copy()
    if set(generated_train.path) & set(generated_valid.path):
        raise ValueError("generated train/validation path overlap")
    for column in ("split_group_id", "near_duplicate_group", "original_id", "speaker_id"):
        overlap = _identifiers(generated_train, column) & _identifiers(generated_valid, column)
        if overlap:
            raise ValueError(f"generated train/validation {column} overlap: {sorted(overlap)[:5]}")
    if _identifiers(generated_train, "generator_family") & _identifiers(generated_valid, "generator_family"):
        raise ValueError("generator families must be disjoint")
    # Column union preserves every existing field and all provenance fields.
    columns = list(dict.fromkeys([*base.columns, *generated.columns]))
    candidate_train = pd.concat([base.reindex(columns=columns),
                                 generated_train.reindex(columns=columns)], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_train.to_csv(output_dir / "train.csv", index=False)
    generated_valid.reindex(columns=columns).to_csv(output_dir / "stress_unseen_fake.csv", index=False)
    risk_components = []
    if risk_report_path is not None and risk_report_path.is_file():
        risk_payload = json.loads(risk_report_path.read_text(encoding="utf-8"))
        risk_components = [str(row.get("component")) for row in risk_payload.get("components", [])
                           if str(row.get("risk", "")).upper() == "HIGH"]
    training_authorized = not risk_components
    report = {
        "status": "PREPARED_NOT_TRAINED", "base_train_rows": int(len(base)),
        "generated_train_rows": int(len(generated_train)),
        "candidate_train_rows": int(len(candidate_train)),
        "unseen_fake_stress_rows": int(len(generated_valid)),
        "selected_v7_splits_modified": False,
        "training_authorized": training_authorized,
        "adoption_status": ("REJECT_CURRENT_DATASET_HIGH_SOURCE_FINGERPRINT"
                            if risk_components else "PENDING_CONTROLLED_VALIDATION"),
        "high_risk_components": risk_components,
        "warning": "stress_unseen_fake contains only synthetic positives; report recall/FNR, not EER.",
    }
    (output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default="data/splits/train.csv")
    parser.add_argument("--generated", default="data/generated_v8/manifest.csv")
    parser.add_argument("--output-dir", default="data/splits_v8_candidate")
    parser.add_argument("--risk-report", default="experiments/v8/domain_risk_report.json")
    args = parser.parse_args()
    report = prepare(ROOT / args.base_train, ROOT / args.generated, ROOT / args.output_dir,
                     ROOT / args.risk_report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
