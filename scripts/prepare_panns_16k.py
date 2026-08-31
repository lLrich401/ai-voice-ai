#!/usr/bin/env python3
"""Download, verify and strip the official 16 kHz PANNs checkpoint.

This is a one-time submission-build step. Inference remains fully offline.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
import urllib.request

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "Cnn14_16k_mAP=0.438.pth"
URL = "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
UPSTREAM_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
PACKAGED_SHA256 = "eee61e89d4ef120bfe0e900f0fb9e4814a2597bbd1f3bf8e149868a7d508bc10"


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    output = ROOT / "model/panns" / NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and digest(output, "sha256") == PACKAGED_SHA256:
        print(f"Verified existing {output}")
        return
    download = output.with_suffix(".upstream.download")
    urllib.request.urlretrieve(URL, download)
    if digest(download, "md5") != UPSTREAM_MD5:
        raise RuntimeError("Official PANNs download MD5 mismatch")
    checkpoint = torch.load(download, map_location="cpu", weights_only=False)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or len(state) != 84:
        raise RuntimeError("Unexpected official PANNs checkpoint structure")
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        packaged = pathlib.Path(directory) / NAME
        torch.save(state, packaged)
        if digest(packaged, "sha256") != PACKAGED_SHA256:
            raise RuntimeError("Normalized PANNs SHA256 mismatch")
        os.replace(packaged, output)
    download.unlink(missing_ok=True)
    print(f"Prepared verified offline checkpoint {output}")


if __name__ == "__main__":
    main()
