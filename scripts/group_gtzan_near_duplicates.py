#!/usr/bin/env python3
"""Group decoded near-identical GTZAN recordings before any data split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_near_duplicates import spectral_fingerprint
from src.preprocess import load_audio


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def normalized_wave_correlation(left, right):
    size = min(len(left), len(right))
    if size == 0 or abs(len(left) - len(right)) / size > 0.001:
        return 0.0
    left = np.asarray(left[:size], dtype=np.float64)
    right = np.asarray(right[:size], dtype=np.float64)
    left -= left.mean(); right -= right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom else 0.0


def group_manifest(manifest_path, report_path, fingerprint_threshold=0.99999,
                   correlation_threshold=0.999):
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    mask = frame["source"].astype(str).str.lower().eq("gtzan_real_v2")
    indices = frame.index[mask].tolist()
    if len(indices) < 980:
        raise RuntimeError(f"Expected repaired GTZAN data, found only {len(indices)} rows")

    fingerprints = []
    for index in indices:
        wave, sr = load_audio(frame.at[index, "path"], target_sr=16000)
        fingerprint, _ = spectral_fingerprint(wave, sr=sr)
        fingerprints.append(fingerprint)
    matrix = np.stack(fingerprints)
    similarities = matrix @ matrix.T
    union = UnionFind(len(indices))
    accepted, rejected = [], []
    for left in range(len(indices)):
        for right in np.flatnonzero(similarities[left, left + 1:] >= fingerprint_threshold) + left + 1:
            left_wave, _ = load_audio(frame.at[indices[left], "path"], target_sr=16000)
            right_wave, _ = load_audio(frame.at[indices[right], "path"], target_sr=16000)
            correlation = normalized_wave_correlation(left_wave, right_wave)
            detail = {
                "left": str(frame.at[indices[left], "path"]),
                "right": str(frame.at[indices[right], "path"]),
                "fingerprint_similarity": float(similarities[left, right]),
                "waveform_correlation": correlation,
            }
            if correlation >= correlation_threshold:
                union.union(left, right)
                accepted.append(detail)
            else:
                rejected.append(detail)

    clusters = {}
    for local_index in range(len(indices)):
        clusters.setdefault(union.find(local_index), []).append(local_index)
    duplicate_clusters = [members for members in clusters.values() if len(members) > 1]
    for members in duplicate_clusters:
        paths = sorted(str(frame.at[indices[i], "path"]) for i in members)
        group_hash = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:20]
        group_id = f"gtzan_near::{group_hash}"
        for local_index in members:
            frame.at[indices[local_index], "split_group_id"] = group_id
    frame.to_csv(manifest_path, index=False)

    report = {
        "status": "PASS",
        "gtzan_rows": len(indices),
        "fingerprint_threshold": fingerprint_threshold,
        "waveform_correlation_threshold": correlation_threshold,
        "accepted_pairs": accepted,
        "rejected_fingerprint_candidates": rejected,
        "duplicate_clusters": len(duplicate_clusters),
        "rows_grouped": int(sum(len(group) for group in duplicate_clusters)),
        "policy": "content-only audit before splitting; labels and scores unused",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--report", default="experiments/gtzan_near_duplicate_groups.json")
    args = parser.parse_args()
    print(json.dumps(group_manifest(args.manifest, args.report), indent=2))


if __name__ == "__main__":
    main()
