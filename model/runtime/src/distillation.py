"""Leakage-safe utilities for V12 specialist knowledge distillation."""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def freeze_teacher(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def teacher_is_frozen(model: torch.nn.Module) -> bool:
    return not model.training and all(not parameter.requires_grad for parameter in model.parameters())


def soft_bernoulli_distillation(student_logits: torch.Tensor,
                                teacher_logits: torch.Tensor,
                                temperature: float = 1.0,
                                mask: torch.Tensor | None = None) -> torch.Tensor:
    """Temperature-scaled soft BCE for one binary head."""
    temperature = float(temperature)
    if temperature not in (1.0, 2.0):
        raise ValueError("V12 only permits distillation temperature 1 or 2")
    targets = torch.sigmoid(teacher_logits.detach() / temperature)
    if not torch.isfinite(targets).all():
        raise FloatingPointError("teacher probabilities are not finite")
    losses = F.binary_cross_entropy_with_logits(
        student_logits / temperature, targets, reduction="none") * (temperature ** 2)
    if mask is None:
        return losses.mean()
    mask = mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def v7_retention_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor],
                      labels: torch.Tensor, task: str,
                      temperature: float = 1.0) -> torch.Tensor:
    """Preserve the selected V7 component, FILE, and presence behavior."""
    component = f"{task}_fake"
    presence = f"{task}_present"
    present_index = 3 if task == "voice" else 4
    present_mask = labels[:, present_index]
    terms = (
        soft_bernoulli_distillation(student["file_fake"], teacher["file_fake"],
                                    temperature, present_mask),
        soft_bernoulli_distillation(student[component], teacher[component],
                                    temperature, present_mask),
        soft_bernoulli_distillation(student[presence], teacher[presence], temperature),
    )
    return 0.4 * terms[0] + 0.4 * terms[1] + 0.2 * terms[2]


def component_teacher_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor],
                           labels: torch.Tensor, task: str,
                           temperature: float = 1.0) -> torch.Tensor:
    """Distill only the strong candidate component head, never its FILE head."""
    component = f"{task}_fake"
    present_index = 3 if task == "voice" else 4
    return soft_bernoulli_distillation(
        student[component], teacher[component], temperature, labels[:, present_index])


def paired_margin_ranking_loss(scores: torch.Tensor, labels: torch.Tensor,
                               paths: list[str] | tuple[str, ...],
                               path_to_group: dict[str, str], margin: float = 0.1) -> torch.Tensor:
    groups: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: {0: [], 1: []})
    for score, label, path in zip(scores, labels, paths):
        group = path_to_group.get(str(path))
        if group is not None:
            groups[group][int(label.item() > 0.5)].append(score)
    losses = []
    for values in groups.values():
        if values[0] and values[1]:
            real = torch.stack(values[0]).mean()
            fake = torch.stack(values[1]).mean()
            losses.append(torch.relu(real - fake + float(margin)))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def source_balanced_weights(frame: pd.DataFrame, task: str) -> np.ndarray:
    """Balance component label, source and generator instead of dataset size."""
    present = frame[f"{task}_present"].astype(int).to_numpy()
    label = frame[f"{task}_fake"].astype(int).to_numpy()
    source = frame.get("source", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    generator = frame.get("generator", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    domain = (source + "::" + generator).to_numpy()
    bucket = np.where(present == 0, "absent", np.where(label == 1, "fake", "real"))
    bucket_mass = {"real": 0.4, "fake": 0.4, "absent": 0.2}
    weights = np.zeros(len(frame), dtype=np.float64)
    for name, mass in bucket_mass.items():
        mask = bucket == name
        if not mask.any():
            continue
        domains = np.unique(domain[mask])
        for current in domains:
            domain_mask = mask & (domain == current)
            weights[domain_mask] = mass / len(domains) / domain_mask.sum()
    if weights.sum() <= 0:
        raise RuntimeError("source-balanced sampler produced zero mass")
    return weights / weights.sum()


class PairedGroupBatchSampler(Sampler[list[int]]):
    """Place real/fake examples from the same TRAIN group in each batch."""

    def __init__(self, frame: pd.DataFrame, task: str, batch_size: int,
                 seed: int = 20260901) -> None:
        if "data_role" in frame and not frame["data_role"].astype(str).str.startswith("train").all():
            raise ValueError("paired sampler accepts TRAIN rows only")
        self.frame = frame.reset_index(drop=True)
        self.task = task
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        component = f"{task}_fake"
        present = f"{task}_present"
        groups: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
        for index, row in self.frame.iterrows():
            if int(row[present]) != 1:
                continue
            group = str(row.get("split_group_id", ""))
            groups[group][int(row[component])].append(index)
        self.pairs = [(values[0], values[1]) for values in groups.values()
                      if values[0] and values[1]]
        if not self.pairs:
            raise ValueError(f"no paired real/fake {task} groups in TRAIN")
        self.weights = source_balanced_weights(self.frame, task)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.frame) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pairs = list(self.pairs)
        rng.shuffle(pairs)
        pair_cursor = 0
        for _ in range(len(self)):
            batch: list[int] = []
            pair_target = max(1, self.batch_size // 4)
            for _ in range(pair_target):
                real, fake = pairs[pair_cursor % len(pairs)]
                pair_cursor += 1
                batch.extend((int(rng.choice(real)), int(rng.choice(fake))))
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                batch.extend(rng.choice(len(self.frame), size=remaining,
                                        replace=True, p=self.weights).tolist())
            random.Random(self.seed + self.epoch + pair_cursor).shuffle(batch)
            yield batch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            source = value.detach().cpu()
            if source.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(source)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}


class V12CachedWaveDataset(Dataset):
    """Memory-mapped deterministic 4-second crops with stochastic channel augmentation."""

    def __init__(self, frame: pd.DataFrame, wave_path, metadata: dict,
                 task: str, augmentation_profile: str = "voice_channel_v9") -> None:
        self.frame = frame.reset_index(drop=True)
        self.waves = np.load(wave_path, mmap_mode="r")
        self.task = str(task)
        self.augmentation_profile = str(augmentation_profile)
        if self.waves.shape != (len(self.frame), 64_000):
            raise RuntimeError(
                f"V12 wave cache shape mismatch: {self.waves.shape} vs {(len(self.frame), 64000)}")
        if int(metadata.get("rows", -1)) != len(self.frame):
            raise RuntimeError("V12 wave cache metadata row mismatch")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        from .augment import AugmentationPipeline
        row = self.frame.iloc[index]
        wave = np.array(self.waves[index], dtype=np.float32, copy=True)
        wave = AugmentationPipeline(
            sr=16_000, is_training=True, profile=self.augmentation_profile)(wave)
        labels = torch.tensor([row.get(head, 0) for head in HEADS], dtype=torch.float32)
        return torch.from_numpy(np.ascontiguousarray(wave)).float(), labels, str(row["path"]), index
