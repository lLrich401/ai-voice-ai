"""Compact, representation-diverse Music forensic research models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio


def constant_q_filterbank(sample_rate: int = 16_000, n_fft: int = 2048,
                          n_bins: int = 72, fmin: float = 125.0,
                          fmax: float = 7_900.0) -> torch.Tensor:
    """Return deterministic triangular filters with constant relative bandwidth."""
    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    centers = torch.exp(torch.linspace(math.log(fmin), math.log(fmax), n_bins))
    log_centers = torch.log(centers)
    edges = torch.empty(n_bins + 1)
    edges[1:-1] = torch.exp((log_centers[:-1] + log_centers[1:]) / 2)
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    filters = torch.zeros(n_bins, len(frequencies))
    for index, center in enumerate(centers):
        left, right = edges[index], edges[index + 1]
        rising = (frequencies - left) / max(float(center - left), 1e-6)
        falling = (right - frequencies) / max(float(right - center), 1e-6)
        band = torch.minimum(rising, falling).clamp(0.0, 1.0)
        if not torch.any(band > 0):
            band[torch.argmin(torch.abs(frequencies - center))] = 1.0
        filters[index] = band / band.sum().clamp_min(1e-8)
    return filters


class _CompactSpectralEncoder(nn.Module):
    def __init__(self, channels: int = 16, embedding_dim: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels), nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2), nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4), nn.SiLU(),
            nn.Conv2d(channels * 4, channels * 4, 3, padding=1),
            nn.BatchNorm2d(channels * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * 4, embedding_dim), nn.SiLU(),
            nn.Dropout(0.2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.projection(self.network(feature.unsqueeze(1)))


class MusicForensicDualBranch(nn.Module):
    """Log-mel plus CQT-style log-frequency forensic detector."""

    representation = "log_mel_plus_stft_constant_q_dual_branch_v1"

    def __init__(self, channels: int = 16, embedding_dim: int = 96) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000, n_fft=1024, win_length=1024, hop_length=160,
            n_mels=96, f_min=30.0, f_max=8_000.0, power=2.0)
        self.register_buffer("cqt_filterbank", constant_q_filterbank(), persistent=True)
        self.register_buffer("cqt_window", torch.hann_window(2048), persistent=False)
        self.mel_encoder = _CompactSpectralEncoder(channels, embedding_dim)
        self.cqt_encoder = _CompactSpectralEncoder(channels, embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.SiLU(), nn.Dropout(0.25))
        self.music_fake = nn.Linear(128, 1)
        self.file_fake = nn.Linear(128, 1)

    @staticmethod
    def _standardize(feature: torch.Tensor) -> torch.Tensor:
        mean = feature.mean(dim=(-2, -1), keepdim=True)
        std = feature.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (feature - mean) / std

    def features(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = torch.log1p(self.mel(wave).clamp_min(0.0))
        spectrum = torch.stft(
            wave, n_fft=2048, hop_length=320, win_length=2048,
            window=self.cqt_window, center=True, return_complex=True)
        power = spectrum.abs().square()
        cqt = torch.einsum("kf,bft->bkt", self.cqt_filterbank, power)
        cqt = torch.log1p(cqt.clamp_min(0.0))
        return self._standardize(mel), self._standardize(cqt)

    def forward(self, wave: torch.Tensor) -> dict[str, torch.Tensor]:
        mel, cqt = self.features(wave)
        embedding = self.fusion(torch.cat(
            [self.mel_encoder(mel), self.cqt_encoder(cqt)], dim=1))
        return {
            "music_fake": self.music_fake(embedding).squeeze(-1),
            "file_fake": self.file_fake(embedding).squeeze(-1),
        }


class PANNsForensicHead(nn.Module):
    """Small Stage-A head trained over frozen 2048-dimensional PANNs embeddings."""

    representation = "official_panns_16k_frozen_embedding_v1"

    def __init__(self, input_dim: int = 2048) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.network(embedding)
        return {"music_fake": output[:, 0], "file_fake": output[:, 1]}
