#!/usr/bin/env python3
"""Add explicit license/provenance fields without changing labels or splits."""
from __future__ import annotations

import argparse
import hashlib
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROVENANCE = {
    ("librispeech_dev", "*"): {
        "source_url": "https://www.openslr.org/12",
        "version": "dev-clean; historical local snapshot",
        "license": "CC BY 4.0",
        "allowed_for_competition": "YES",
        "redistribution_allowed": "YES_WITH_ATTRIBUTION",
        "commercial_restriction": "NO",
    },
    ("asvspoof2019", "*"): {
        "source_url": "https://www.asvspoof.org/",
        "version": "ASVspoof 2019 LA; mirror snapshot not recorded",
        "license": "Custom corpus usage agreement",
        "allowed_for_competition": "REVIEW_REQUIRED",
        "redistribution_allowed": "NO_UNTIL_VERIFIED",
        "commercial_restriction": "TERMS_REVIEW_REQUIRED",
    },
    ("wavefake_ajay", "*"): {
        "source_url": "https://doi.org/10.5281/zenodo.5642694",
        "version": "WaveFake mirror snapshot not recorded",
        "license": "CC BY-SA 4.0 upstream; source restrictions apply",
        "allowed_for_competition": "REVIEW_REQUIRED",
        "redistribution_allowed": "NO_UNTIL_VERIFIED",
        "commercial_restriction": "SOURCE_TERMS_REVIEW_REQUIRED",
    },
    ("gtzan_real_v2", "*"): {
        "source_url": "https://huggingface.co/datasets/sanchit-gandhi/gtzan",
        "version": "repaired local parquet snapshot; 985 unique tracks",
        "license": "UNKNOWN_IN_INSPECTED_MIRROR",
        "allowed_for_competition": "REVIEW_REQUIRED",
        "redistribution_allowed": "NO_UNTIL_VERIFIED",
        "commercial_restriction": "UNKNOWN",
    },
    ("fake_music_generated", "MusicGen"): {
        "source_url": "https://huggingface.co/facebook/musicgen-large",
        "version": "historical output run; prompts/commit not recorded",
        "license": "CC BY-NC 4.0 model; output provenance incomplete",
        "allowed_for_competition": "REVIEW_REQUIRED",
        "redistribution_allowed": "NO_UNTIL_VERIFIED",
        "commercial_restriction": "YES_MODEL_WEIGHTS",
    },
    ("fake_music_generated", "AudioLDM2"): {
        "source_url": "https://audioldm.github.io/audioldm2/",
        "version": "historical output run; exact checkpoint not recorded",
        "license": "UNKNOWN_EXACT_CHECKPOINT_AND_OUTPUT_TERMS",
        "allowed_for_competition": "REVIEW_REQUIRED",
        "redistribution_allowed": "NO_UNTIL_VERIFIED",
        "commercial_restriction": "UNKNOWN",
    },
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_for(row):
    key = (str(row["source"]), str(row["generator"]))
    value = PROVENANCE.get(key) or PROVENANCE.get((key[0], "*"))
    if value is None:
        raise RuntimeError(f"No provenance policy for source/generator={key}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    args = parser.parse_args()
    path = ROOT / args.manifest
    frame = pd.read_csv(path)
    metadata = [provenance_for(row) for _, row in frame.iterrows()]
    for column in ("source_url", "version", "license", "allowed_for_competition",
                   "redistribution_allowed", "commercial_restriction"):
        frame[column] = [item[column] for item in metadata]
    frame["dataset_name"] = frame["dataset"].astype(str)
    frame["content_hash"] = [sha256(ROOT / value) for value in frame["path"].astype(str)]
    frame["near_duplicate_group"] = frame["split_group_id"].astype(str)
    required = (
        "dataset_name", "source_url", "version", "license",
        "allowed_for_competition", "redistribution_allowed",
        "commercial_restriction", "original_id", "speaker_id", "generator",
        "content_hash", "near_duplicate_group", "split_group_id",
    )
    if frame[list(required)].isna().any().any() or (frame[list(required)].astype(str) == "").any().any():
        raise RuntimeError("Provenance enrichment left a required field empty")
    frame.to_csv(path, index=False)
    print(frame["allowed_for_competition"].value_counts().to_string())
    print(f"Wrote {len(frame)} provenance-complete rows to {path}")


if __name__ == "__main__":
    main()
