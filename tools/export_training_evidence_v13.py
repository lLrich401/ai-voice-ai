#!/usr/bin/env python3
"""Export compact V13 training provenance without copying training audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree_files(source: pathlib.Path, destination: pathlib.Path) -> list[str]:
    copied = []
    if not source.exists():
        return copied
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target.relative_to(ROOT).as_posix())
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits_v13/train_approved.csv")
    parser.add_argument("--output", default="training_evidence_v13")
    args = parser.parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    output = (ROOT / args.output).resolve()
    frame = pd.read_csv(manifest_path)
    if not frame.competition_use_status.eq("APPROVED").all():
        raise RuntimeError("training evidence must not silently include unapproved rows")
    output.mkdir(parents=True, exist_ok=True)

    evidence_columns = [
        "path", "sha256", "source", "source_url", "license", "generator",
        "generator_version", "language", "speaker", "track", "content_group",
        "split_group", "augmentation", "seed", "competition_use_status",
    ]
    for column in evidence_columns:
        if column not in frame:
            frame[column] = "unknown"
    frame[evidence_columns].to_csv(output / "used_training_files.csv", index=False)
    frame.to_csv(output / "manifest.csv", index=False)

    copied = []
    copied += copy_tree_files(ROOT / "configs/v13", output / "configs")
    copied += copy_tree_files(ROOT / "data/provenance", output / "provenance")
    copied += copy_tree_files(ROOT / "data/licenses", output / "licenses")
    licenses = output / "licenses"
    provenance = output / "provenance"
    licenses.mkdir(exist_ok=True)
    provenance.mkdir(exist_ok=True)
    frame[["source", "source_url", "license", "competition_use_status"]].drop_duplicates().to_csv(
        licenses / "dataset_terms.csv", index=False)
    frame[["source", "source_url", "generator", "generator_version"]].drop_duplicates().to_csv(
        provenance / "sources_and_generators.csv", index=False)
    generation_scripts = output / "generation_scripts"
    generation_scripts.mkdir(exist_ok=True)
    for relative in ("scripts/generate_procedural_dataset_v8.py", "scripts/prepare_dataset_v13.py"):
        source = ROOT / relative
        if source.exists():
            shutil.copy2(source, generation_scripts / source.name)

    selected = json.loads((ROOT / "archive/pre_v13_selected/artifact_manifest.json").read_text(
        encoding="utf-8"))
    (output / "checkpoint_hashes.json").write_text(json.dumps({
        "status": "PRE_V13_SELECTED_ONLY; NO V13 CHECKPOINT TRAINED",
        "artifacts": selected["selected_artifacts"],
    }, indent=2) + "\n", encoding="utf-8")
    dataset_files = sorted(
        path for path in (ROOT / "data/splits_v13").glob("*.csv")
        if path.name != "final_holdout_v13.csv")
    (output / "dataset_hashes.json").write_text(json.dumps({
        path.relative_to(ROOT).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in dataset_files
    }, indent=2) + "\n", encoding="utf-8")
    (output / "training_commands.txt").write_text(
        "# V13 training intentionally NOT RUN: source-shortcut gate failed.\n"
        "python scripts/prepare_dataset_v13.py\n"
        "python tools/audit_source_shortcut_v13.py --manifest data/splits_v13/train_approved.csv --limit 0\n",
        encoding="utf-8")
    (output / "README_REPRODUCE.md").write_text(
        "# V13 training evidence\n\n"
        "This directory contains manifests, hashes, configuration, and generation tooling only. "
        "It intentionally excludes audio and hidden DACON test data. V13 model training was not "
        "started because the pre-registered source-shortcut AUC gate failed. Rebuild the pilot, "
        "run the shortcut audit, and train only after the audit passes. FINAL_HOLDOUT_V13 remains "
        "sealed until a candidate is locked.\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "EXPORTED_NO_V13_TRAINING_RUN",
        "rows": len(frame),
        "manifest_sha256": sha256(output / "manifest.csv"),
        "copied_metadata_files": len(copied),
        "audio_files_copied": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
