"""Upstream-compatible PANNs CNN14 presence inference.

The bundled checkpoint is the official 16 kHz CNN14 model. Its frontend is
part of the learned model: changing FFT/hop/log-mel or skipping ``bn0``
invalidates the pretrained weights even if convolutional coverage is high.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


@dataclass(frozen=True)
class PANNsFrontendConfig:
    sample_rate: int = 16_000
    window_size: int = 512
    hop_size: int = 160
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 8_000


OFFICIAL_16K_CONFIG = PANNsFrontendConfig()
PANNs_CHECKPOINT_NAME = "Cnn14_16k_mAP=0.438.pth"
# State-dict-only derivative of official Zenodo artifact MD5
# 362fc5ff18f1d6ad2f6d464b45893f2c; training sampler removed.
PANNs_CHECKPOINT_SHA256 = "eee61e89d4ef120bfe0e900f0fb9e4814a2597bbd1f3bf8e149868a7d508bc10"


def _init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(batch_norm: nn.Module) -> None:
    batch_norm.bias.data.fill_(0.0)
    batch_norm.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size=(2, 2), pool_type="avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg+max":
            return F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        raise ValueError(f"Invalid pool type: {pool_type}")


class Cnn14(nn.Module):
    """CNN14 with the exact official 16 kHz torchlibrosa frontend."""

    def __init__(self, config: PANNsFrontendConfig = OFFICIAL_16K_CONFIG, classes_num: int = 527) -> None:
        super().__init__()
        self.frontend_config = config
        self.spectrogram_extractor = Spectrogram(
            n_fft=config.window_size, hop_length=config.hop_size,
            win_length=config.window_size, window="hann", center=True,
            pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate, n_fft=config.window_size,
            n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(config.mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    def forward(self, input: torch.Tensor, mixup_lambda=None) -> dict[str, torch.Tensor]:
        del mixup_lambda
        x = self.logmel_extractor(self.spectrogram_extractor(input))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for block, pool_size in (
            (self.conv_block1, (2, 2)), (self.conv_block2, (2, 2)),
            (self.conv_block3, (2, 2)), (self.conv_block4, (2, 2)),
            (self.conv_block5, (2, 2)), (self.conv_block6, (1, 1))):
            x = block(x, pool_size=pool_size, pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        return {"clipwise_output": torch.sigmoid(self.fc_audioset(x)), "embedding": embedding}


PANNsCNN14 = Cnn14


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PANNsPresenceWrapper(nn.Module):
    """Map pretrained AudioSet CNN14 tags to voice/music presence."""

    SPEECH_INDICES = (0, 1, 2, 3, 4)
    MUSIC_INDICES = (137, 138, 139, 140, 141, 142, 143, 144, 145)

    def __init__(self, sample_rate: int = 16_000, use_pretrained: bool = True,
                 checkpoint_path: str | pathlib.Path | None = None,
                 verify_sha256: bool = True) -> None:
        super().__init__()
        if sample_rate != OFFICIAL_16K_CONFIG.sample_rate:
            raise ValueError("The bundled PANNs checkpoint only supports 16 kHz input")
        self.frontend_config = OFFICIAL_16K_CONFIG
        self.panns = Cnn14(self.frontend_config, classes_num=527)
        self.pretrained_loaded = False
        self.load_stats: dict[str, object] | None = None
        self.checkpoint_path: pathlib.Path | None = None
        if use_pretrained:
            self._load_pretrained(checkpoint_path, verify_sha256)

    @staticmethod
    def checkpoint_candidates() -> tuple[pathlib.Path, ...]:
        module_path = pathlib.Path(__file__).resolve()
        return (
            pathlib.Path("model/panns") / PANNs_CHECKPOINT_NAME,
            module_path.parents[2] / "model" / "panns" / PANNs_CHECKPOINT_NAME,
            module_path.parents[3] / "panns" / PANNs_CHECKPOINT_NAME,
        )

    def _load_pretrained(self, checkpoint_path, verify_sha256: bool) -> None:
        candidates = ((pathlib.Path(checkpoint_path),) if checkpoint_path else self.checkpoint_candidates())
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Required PANNs checkpoint not found: model/panns/{PANNs_CHECKPOINT_NAME}")
        if verify_sha256:
            actual_sha = _sha256(path)
            if actual_sha != PANNs_CHECKPOINT_SHA256:
                raise RuntimeError(f"PANNs checkpoint SHA256 mismatch: expected {PANNs_CHECKPOINT_SHA256}, got {actual_sha}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError("PANNs checkpoint must contain a state dict")
        incompatible = self.panns.load_state_dict(checkpoint, strict=True)
        self.load_stats = {
            "checkpoint_keys": len(checkpoint), "loaded_keys": len(checkpoint),
            "key_coverage": 1.0, "element_coverage": 1.0,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "frontend": asdict(self.frontend_config),
            "checkpoint_sha256": PANNs_CHECKPOINT_SHA256,
        }
        self.checkpoint_path = path.resolve()
        self.pretrained_loaded = True

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.pretrained_loaded:
            raise RuntimeError("PANNs pretrained checkpoint is mandatory")
        output = self.panns(wav)
        clipwise = output["clipwise_output"]
        return {
            "voice_present": clipwise[:, self.SPEECH_INDICES].amax(dim=1),
            "music_present": clipwise[:, self.MUSIC_INDICES].amax(dim=1),
            "embedding": output["embedding"], "clipwise": clipwise,
        }

    def predict_presence(self, wav: torch.Tensor, device="cpu"):
        self.eval()
        with torch.inference_mode():
            wav = wav.to(device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            output = self.forward(wav)
            return output["voice_present"].cpu().numpy(), output["music_present"].cpu().numpy()
