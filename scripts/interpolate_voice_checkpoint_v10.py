#!/usr/bin/env python3
"""Create strict single-model interpolation candidates; never touches holdout data."""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import torch


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="model/best.pt")
    parser.add_argument("--finetuned", default="model/candidates/voice_spec_cnn_v10_finetune.pt")
    parser.add_argument("--output_dir", default="model/candidates")
    parser.add_argument("--alphas", default="0.25,0.5,0.75")
    args = parser.parse_args()
    base_path, tuned_path = pathlib.Path(args.base), pathlib.Path(args.finetuned)
    base = torch.load(base_path, map_location="cpu")
    tuned = torch.load(tuned_path, map_location="cpu")
    required = ("task", "backbone", "base_channels", "sample_rate", "label_heads")
    mismatch = {key: (base.get(key), tuned.get(key)) for key in required
                if base.get(key) != tuned.get(key)}
    if mismatch:
        raise ValueError(f"checkpoint metadata mismatch: {mismatch}")
    if base["model"].keys() != tuned["model"].keys():
        raise ValueError("checkpoint state keys differ")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for alpha in (float(value) for value in args.alphas.split(",")):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        state = {}
        for key, base_value in base["model"].items():
            tuned_value = tuned["model"][key]
            if base_value.dtype.is_floating_point:
                state[key] = torch.lerp(base_value, tuned_value, alpha)
            else:
                state[key] = tuned_value.clone() if alpha >= 0.5 else base_value.clone()
        payload = dict(base)
        payload.update({
            "model": state,
            "score": None,
            "selection_score": None,
            "augmentation_profile": "weight_interpolation_validation_candidate",
            "interpolation_alpha": alpha,
            "interpolation_base_sha256": sha256(base_path),
            "interpolation_finetuned_sha256": sha256(tuned_path),
            "final_holdout": "NOT RUN",
        })
        suffix = str(alpha).replace(".", "p")
        output = output_dir / f"voice_spec_cnn_v10_interp_{suffix}.pt"
        torch.save(payload, output)
        print(f"{alpha:.2f} {output} {sha256(output)}")


if __name__ == "__main__":
    main()
