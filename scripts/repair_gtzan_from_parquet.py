#!/usr/bin/env python3
"""Recover the 1,000 unique GTZAN tracks from cached Hugging Face parquet shards.

The operation is deliberately fail-closed: audio is written to a new directory,
all hashes are checked, and the manifest is replaced only after every validation
passes. Existing GTZAN audio is never deleted or overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pyarrow.parquet as pq


EXPECTED_INPUT_TRACKS = 999
MINIMUM_UNIQUE_TRACKS = 980


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def iter_gtzan_rows(shards):
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=16, columns=["file", "audio", "genre"]):
            for row in batch.to_pylist():
                audio = row["audio"] or {}
                payload = audio.get("bytes")
                if not payload:
                    raise RuntimeError(f"Missing embedded audio bytes in {shard}: {row['file']}")
                yield str(row["file"]), bytes(payload), int(row["genre"])


def repair(parquet_dir, output_dir, manifest_path, report_path):
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    report_path = Path(report_path)
    shards = sorted(parquet_dir.glob("*.parquet"))
    if len(shards) != 3:
        raise RuntimeError(f"Expected 3 GTZAN parquet shards, found {len(shards)} in {parquet_dir}")
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    staging = output_dir.with_name(output_dir.name + "_staging")
    if staging.exists():
        raise RuntimeError(f"Remove stale staging directory before retrying: {staging}")
    staging.mkdir(parents=True)
    records, hashes, names, duplicates = [], set(), set(), []
    input_rows = 0
    try:
        for source_name, payload, genre in iter_gtzan_rows(shards):
            input_rows += 1
            filename = Path(source_name).name
            if filename in names:
                raise RuntimeError(f"Duplicate GTZAN filename: {filename}")
            digest = sha256_bytes(payload)
            if digest in hashes:
                duplicates.append({"filename": filename, "sha256": digest})
                continue
            names.add(filename)
            hashes.add(digest)
            destination = staging / filename
            destination.write_bytes(payload)
            relative_path = (output_dir / filename).as_posix()
            stem = Path(filename).stem
            original_id = f"gtzan_v2_{stem}"
            records.append({
                "path": relative_path,
                "file_fake": 0,
                "voice_fake": 0,
                "music_fake": 0,
                "voice_present": 0,
                "music_present": 1,
                "speaker_id": f"gtzan_{genre}_{stem}",
                "generator": f"GTZAN_genre_{genre}",
                "source": "gtzan_real_v2",
                "dataset": "gtzan_real",
                "hf_id": "sanchit-gandhi/gtzan",
                "original_id": original_id,
                "split_group_id": f"gtzan_sha256::{digest}",
            })
        if input_rows != EXPECTED_INPUT_TRACKS or len(hashes) < MINIMUM_UNIQUE_TRACKS:
            raise RuntimeError(
                f"GTZAN integrity failure: input_rows={input_rows}, unique_hashes={len(hashes)}, "
                f"expected_input={EXPECTED_INPUT_TRACKS}, minimum_unique={MINIMUM_UNIQUE_TRACKS}"
            )

        old = pd.read_csv(manifest_path)
        required = set(records[0])
        missing = required - set(old.columns)
        if missing:
            raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")
        is_old_gtzan = old["source"].astype(str).str.lower().str.startswith("gtzan")
        preserved = old.loc[~is_old_gtzan].copy()
        repaired = pd.concat([preserved, pd.DataFrame(records)], ignore_index=True, sort=False)
        if repaired["path"].astype(str).duplicated().any():
            raise RuntimeError("Repaired manifest contains duplicate paths")

        staging.rename(output_dir)
        temp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        repaired.to_csv(temp_manifest, index=False)
        os.replace(temp_manifest, manifest_path)
        report = {
            "status": "PASS",
            "parquet_shards": [str(p) for p in shards],
            "old_gtzan_rows_replaced": int(is_old_gtzan.sum()),
            "parquet_input_rows": input_rows,
            "new_gtzan_rows": len(records),
            "new_gtzan_unique_sha256": len(hashes),
            "source_duplicate_rows_deduplicated": len(duplicates),
            "source_duplicates": duplicates,
            "preserved_non_gtzan_rows": len(preserved),
            "manifest_rows": len(repaired),
            "output_directory": str(output_dir),
            "policy": "exact source duplicates removed; manifest updated only after integrity checks",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main():
    default_cache = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "datasets--sanchit-gandhi--gtzan" / "snapshots"
        / "4bd857132cb0e731bef3ec68558e7acc0a85f144" / "data"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default=str(default_cache))
    parser.add_argument("--output-dir", default="data/raw/gtzan_unique_v2")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--report", default="experiments/gtzan_repair_report.json")
    args = parser.parse_args()
    report = repair(args.parquet_dir, args.output_dir, args.manifest, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
