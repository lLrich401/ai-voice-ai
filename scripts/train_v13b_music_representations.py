#!/usr/bin/env python3
"""Cheap M1/M2 Music representation probes on non-final V13B splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.manage_v13b_stages import evaluate_gates
from scripts.prepare_dataset_v13b import mixed_rows, partial_rows
from src.dataset import AudioDataset, load_manifest_row_wave
from src.metrics import compute_eer
from src.models.music_forensic import MusicForensicDualBranch, PANNsForensicHead
from src.models.panns import PANNsPresenceWrapper, PANNs_CHECKPOINT_SHA256
from src.train import specialist_sample_weights
from tools.v13_guards import assert_final_holdout_v13b_forbidden

# Import the submission module only after the repository ``src`` package is
# loaded. script.py prepends its bundled runtime path for offline submission.
import script as submission


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment_groups(frame: pd.DataFrame) -> list[list[np.ndarray]]:
    waves = [load_manifest_row_wave(row, sr=16_000, is_training=False,
                                    use_demucs=False, task="music")
             for _, row in frame.iterrows()]
    return [submission.select_aux_segments(wave, policy="high_energy") for wave in waves]


@torch.inference_mode()
def predict_wave_model(model: torch.nn.Module, frame: pd.DataFrame,
                       device: torch.device, batch_size: int = 16) -> dict[str, np.ndarray]:
    groups = segment_groups(frame)
    flat = np.stack([segment for group in groups for segment in group])
    bounds, offset = [], 0
    for group in groups:
        bounds.append((offset, offset + len(group))); offset += len(group)
    output = {"music_fake": [], "file_fake": []}
    for start in range(0, len(flat), batch_size):
        batch = torch.from_numpy(flat[start:start + batch_size]).float().to(device)
        logits = model(batch)
        for head in output:
            values = torch.sigmoid(logits[head]).detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"M1 non-finite output in {head}")
            output[head].extend(values.tolist())
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text())
    return {head: np.asarray([
        submission.aggregate_head_predictions(np.asarray(values[left:right]), head, weights)
        for left, right in bounds]) for head, values in output.items()}


@torch.inference_mode()
def panns_embeddings(groups: list[list[np.ndarray]], panns: PANNsPresenceWrapper,
                     device: torch.device, batch_size: int = 4) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    flat = np.stack([segment for group in groups for segment in group])
    bounds, offset = [], 0
    for group in groups:
        bounds.append((offset, offset + len(group))); offset += len(group)
    embeddings = []
    for start in range(0, len(flat), batch_size):
        batch = torch.from_numpy(flat[start:start + batch_size]).float().to(device)
        value = panns(batch)["embedding"]
        if not torch.isfinite(value).all():
            raise RuntimeError("M2 PANNs embedding contains non-finite values")
        embeddings.append(value.cpu())
    return torch.cat(embeddings), bounds


@torch.inference_mode()
def predict_panns_head(head: PANNsForensicHead, panns: PANNsPresenceWrapper,
                       frame: pd.DataFrame, device: torch.device) -> dict[str, np.ndarray]:
    embeddings, bounds = panns_embeddings(segment_groups(frame), panns, device)
    logits = head(embeddings.to(device))
    weights = json.loads((ROOT / "model/fusion_weights.json").read_text())
    result = {}
    for name, tensor in logits.items():
        values = torch.sigmoid(tensor).cpu().numpy()
        result[name] = np.asarray([
            submission.aggregate_head_predictions(values[left:right], name, weights)
            for left, right in bounds])
    return result


def evaluate_predictions(frame: pd.DataFrame, prediction: dict[str, np.ndarray]) -> dict:
    music_mask = frame.music_present.to_numpy(int) == 1
    return {
        "music_eer": compute_eer(frame.music_fake.to_numpy(int)[music_mask],
                                 prediction["music_fake"][music_mask]),
        "standalone_file_eer": compute_eer(frame.file_fake, prediction["file_fake"]),
    }


def partial_mixed_metrics(model_predict, validation: pd.DataFrame) -> dict:
    partial = partial_rows(validation.assign(data_role="paired_core"))
    mixed = mixed_rows(validation.assign(data_role="paired_core"), per_state=12)
    partial_scores = model_predict(partial)["file_fake"]
    mixed_scores = model_predict(mixed)["file_fake"]
    partial_result = {
        "rows": len(partial),
        "file_eer": compute_eer(partial.file_fake, partial_scores),
        "by_ratio": {},
    }
    for ratio, group in partial.groupby("partial_fake_ratio"):
        indices = group.index.to_numpy()
        partial_result["by_ratio"][str(ratio)] = compute_eer(
            group.file_fake, partial_scores[indices])
    state_result = {}
    for state, group in mixed.groupby("mix_state"):
        indices = group.index.to_numpy()
        scores = mixed_scores[indices]
        truth = group.file_fake.to_numpy(int)
        state_result[state] = {
            "rows": len(group), "mean_file_score": float(np.mean(scores)),
            "threshold_0_5_error_rate": float(np.mean((scores >= 0.5).astype(int) != truth)),
            "metric_note": "single-class state; EER unavailable",
        }
    return {
        "scope": "MEASURED_GENERATOR_DISJOINT_ROOTS_WITH_DETERMINISTIC_VIRTUAL_RENDERING",
        "partial": partial_result,
        "mixed": {
            "rows": len(mixed), "file_eer_rr_vs_fake_states": compute_eer(mixed.file_fake, mixed_scores),
            "states": state_result,
        },
    }


def train_m1(train: pd.DataFrame, device: torch.device, epochs: int,
             batch_size: int, seed: int) -> tuple[MusicForensicDualBranch, list[dict], float]:
    model = MusicForensicDualBranch().to(device)
    data = AudioDataset(train, sr=16_000, seg_sec=4.0, is_training=True,
                        use_demucs=False, task="music", device=str(device),
                        augmentation_profile="baseline")
    weights = specialist_sample_weights(train, "music")
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(train), replacement=True)
    loader = DataLoader(data, batch_size=batch_size, sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history, started = [], time.perf_counter()
    for epoch in range(epochs):
        model.train(); total = 0.0; count = 0
        for wave, labels, _ in loader:
            wave, labels = wave.to(device), labels.to(device)
            logits = model(wave)
            present = labels[:, 4] > 0.5
            if not torch.any(present):
                continue
            music_loss = F.binary_cross_entropy_with_logits(
                logits["music_fake"][present], labels[present, 2])
            file_loss = F.binary_cross_entropy_with_logits(logits["file_fake"], labels[:, 0])
            loss = music_loss + 0.25 * file_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(wave); count += len(wave)
        value = total / max(count, 1)
        history.append({"epoch": epoch + 1, "loss": value})
        print(f"M1 epoch {epoch + 1}/{epochs} loss={value:.6f}", flush=True)
    return model.eval(), history, time.perf_counter() - started


def train_m2(train: pd.DataFrame, device: torch.device, epochs: int,
             batch_size: int) -> tuple[PANNsForensicHead, PANNsPresenceWrapper, list[dict], float]:
    panns = PANNsPresenceWrapper(use_pretrained=True).to(device).eval()
    for parameter in panns.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    cache_path = ROOT / "model/candidates/v13b/m2_train_embedding_cache.pt"
    split_sha = hashlib.sha256((ROOT / "data/splits_v13b/train.csv").read_bytes()).hexdigest()
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (cache.get("split_sha256") != split_sha or
                cache.get("panns_sha256") != PANNs_CHECKPOINT_SHA256):
            raise RuntimeError("M2 embedding cache dependency mismatch")
        features, targets = cache["features"], cache["targets"]
    else:
        deterministic = AudioDataset(train, sr=16_000, seg_sec=4.0, is_training=False,
                                     use_demucs=False, task="music", device=str(device))
        loader = DataLoader(deterministic, batch_size=4, shuffle=False, num_workers=0)
        embeddings, labels = [], []
        with torch.inference_mode():
            for wave, target, _ in loader:
                embeddings.append(panns(wave.to(device))["embedding"].cpu())
                labels.append(target)
        features, targets = torch.cat(embeddings), torch.cat(labels)
        torch.save({
            "features": features, "targets": targets, "split_sha256": split_sha,
            "panns_sha256": PANNs_CHECKPOINT_SHA256,
        }, cache_path)
    head = PANNsForensicHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(23674913)
    head_loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size,
                             shuffle=True, generator=generator)
    history = []
    for epoch in range(epochs):
        head.train(); total = 0.0
        for feature, target in head_loader:
            feature, target = feature.to(device), target.to(device)
            logits = head(feature); present = target[:, 4] > 0.5
            music_loss = (F.binary_cross_entropy_with_logits(
                logits["music_fake"][present], target[present, 2])
                if torch.any(present) else logits["music_fake"].sum() * 0.0)
            file_loss = F.binary_cross_entropy_with_logits(logits["file_fake"], target[:, 0])
            loss = music_loss + 0.25 * file_loss
            if not torch.isfinite(loss):
                raise RuntimeError("M2 head loss became non-finite")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(feature)
        history.append({"epoch": epoch + 1, "loss": total / len(features)})
        print(f"M2 head epoch {epoch + 1}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    return head.eval(), panns, history, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("m1", "m2"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23674913)
    args = parser.parse_args(); seed_everything(args.seed)
    train_path = ROOT / "data/splits_v13b/train.csv"
    validation_path = ROOT / "data/splits_v13b/val_generator_disjoint.csv"
    assert_final_holdout_v13b_forbidden(train_path, validation_path)
    train, validation = pd.read_csv(train_path), pd.read_csv(validation_path)
    dataset_report = json.loads((ROOT / "data/splits_v13b/DATASET_V13B.json").read_text())
    shortcut = json.loads((ROOT / "experiments/v13b/source_shortcut_audit.json").read_text())
    policy = json.loads((ROOT / "configs/v13b/selection_policy.json").read_text())
    if not evaluate_gates(dataset_report, shortcut, policy)["exploratory_allowed"]:
        raise RuntimeError("exploratory safety gate is not satisfied")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_dir = ROOT / "model/candidates/v13b"; candidate_dir.mkdir(parents=True, exist_ok=True)
    if args.candidate == "m1":
        epochs = args.epochs or 2
        model, history, training_seconds = train_m1(
            train, device, epochs, args.batch_size, args.seed)
        prediction = lambda frame: predict_wave_model(model, frame, device)
        candidate_path = candidate_dir / "m1_music_logmel_cqt_probe.pt"
        checkpoint = {
            "model": model.state_dict(), "candidate": "M1", "epochs": epochs,
            "representation": model.representation, "candidate_status": "NOT_ADOPTABLE_YET",
            "validation": "generator-disjoint only", "source_disjoint": "NOT MEASURED",
            "final_holdout": "NOT RUN", "history": history,
        }
    else:
        epochs = args.epochs or 8
        model, panns, history, training_seconds = train_m2(
            train, device, epochs, args.batch_size)
        prediction = lambda frame: predict_panns_head(model, panns, frame, device)
        candidate_path = candidate_dir / "m2_music_panns_frozen_probe.pt"
        checkpoint = {
            "model": model.state_dict(), "candidate": "M2", "epochs": epochs,
            "representation": model.representation, "panns_checkpoint_sha256": PANNs_CHECKPOINT_SHA256,
            "backbone_frozen": True, "candidate_status": "NOT_ADOPTABLE_YET",
            "validation": "generator-disjoint only", "source_disjoint": "NOT MEASURED",
            "final_holdout": "NOT RUN", "history": history,
        }
    torch.save(checkpoint, candidate_path)
    evaluation_started = time.perf_counter()
    predictions = prediction(validation)
    metrics = evaluate_predictions(validation, predictions)
    diagnostics = partial_mixed_metrics(prediction, validation)
    baseline_music_eer = 0.3125
    delta_music = baseline_music_eer - metrics["music_eer"]
    signal = ("PROMISING" if metrics["music_eer"] <= 0.20 else
              "STRONG" if metrics["music_eer"] <= 0.25 else
              "CLEAR" if metrics["music_eer"] <= 0.28 else "INSUFFICIENT")
    report = {
        "status": "MEASURED_GENERATOR_DISJOINT_NOT_ADOPTABLE",
        "candidate": args.candidate.upper(), "representation": checkpoint["representation"],
        "checkpoint": candidate_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(candidate_path), "device": str(device),
        "training_rows": len(train), "training_seconds": training_seconds,
        "epochs": epochs, "history": history,
        "generator_disjoint": {**metrics, "rows": len(validation)},
        "partial_mixed": diagnostics,
        "ads_contributions": {
            "delta_file": 0.5 * (0.4242424242424242 - metrics["standalone_file_eer"]),
            "delta_music": 0.3 * delta_music,
            "delta_voice": 0.0,
            "voice_note": "not changed by this Music-only exploratory probe",
        },
        "screening": {
            "baseline_music_eer": baseline_music_eer,
            "candidate_music_eer": metrics["music_eer"],
            "clear_threshold": 0.28, "strong_threshold": 0.25,
            "promising_threshold": 0.20, "signal": signal,
        },
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "source_disjoint": "NOT MEASURED", "final_holdout": "NOT RUN",
        "selected_artifacts_mutated": False, "decision": "KEEP_TEST5",
    }
    output = ROOT / f"experiments/v13b/{args.candidate}_music_representation_probe.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
