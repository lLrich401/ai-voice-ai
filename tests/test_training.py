import numpy as np
import pandas as pd
import pytest
import torch

from src.augment import AugmentationPipeline, rerecording_simulation, signed_linear_quantize
from src.dataset import apply_codec_sim, apply_telephone_sim

from src.train import (
    apply_hard_example_weights, checkpoint_selection_score, masked_multitask_loss,
    specialist_sample_weights,
)


def test_absent_component_labels_do_not_change_multitask_loss():
    logits = torch.zeros((2, 5))
    labels = torch.tensor([
        [0, 0, 0, 0, 1],  # voice label is undefined
        [1, 1, 0, 1, 0],  # music label is undefined
    ], dtype=torch.float32)
    changed = labels.clone()
    changed[0, 1] = 1
    changed[1, 2] = 1
    assert torch.equal(
        masked_multitask_loss(logits, labels),
        masked_multitask_loss(logits, changed),
    )


def test_checkpoint_selection_penalizes_weak_unseen_split():
    good = {name: {"score": 0.8} for name in ("val_a", "val_b", "val_c", "val_d")}
    weak = {name: {"score": 0.8} for name in ("val_a", "val_b", "val_c", "val_d")}
    weak["val_b"] = {"score": 0.2}
    assert checkpoint_selection_score(good) > checkpoint_selection_score(weak)


def test_specialist_checkpoint_uses_its_own_responsibility():
    base = {"score": 0.5, "voice_eer": 0.1, "voice_auc": 0.9,
            "music_eer": 0.4, "music_auc": 0.6}
    voice_better = {name: dict(base) for name in ("val_a", "val_b")}
    music_better = {name: dict(base) for name in ("val_a", "val_b")}
    for metrics in voice_better.values():
        metrics["voice_eer"] = 0.0
    for metrics in music_better.values():
        metrics["music_eer"] = 0.0
    assert checkpoint_selection_score(voice_better, "voice") > checkpoint_selection_score(music_better, "voice")
    assert checkpoint_selection_score(music_better, "music") > checkpoint_selection_score(voice_better, "music")


def test_hard_negative_sampler_rejects_validation_leakage():
    frame = pd.DataFrame({
        "path": ["real.wav", "fake.wav"], "voice_present": [1, 1],
        "music_present": [0, 0], "voice_fake": [0, 1],
        "source": ["train", "train"], "generator": ["real", "fake"],
    })
    base = specialist_sample_weights(frame, "voice")
    leaked = pd.DataFrame({
        "path": frame["path"], "voice_fake_score": [0.9, 0.1],
        "data_role": ["train", "val_b"],
    })
    with pytest.raises(ValueError, match="non-TRAIN"):
        apply_hard_example_weights(frame, base, leaked)
    weighted = apply_hard_example_weights(frame, base, leaked.assign(data_role="train"))
    assert np.isclose(weighted.sum(), 1.0)
    assert np.all(weighted > 0)


def test_rerecording_simulation_preserves_shape_and_range():
    import random
    random.seed(19)
    np.random.seed(19)
    wave = np.sin(np.linspace(0, 250, 32000)).astype(np.float32) * 0.15
    result = rerecording_simulation(wave, p=1.0)
    assert result.shape == wave.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(np.max(np.abs(result))) <= 1.0
    assert not np.array_equal(result, wave)


def test_specialist_sampler_balances_real_against_many_fake_generators():
    frame = pd.DataFrame({
        "path": ["real.wav"] * 2 + [f"fake_{i}.wav" for i in range(8)],
        "voice_present": [1] * 10, "music_present": [0] * 10,
        "voice_fake": [0, 0] + [1] * 8,
        "source": ["paired"] * 10,
        "generator": ["original"] * 2 + [f"generator_{i}" for i in range(8)],
    })
    weights = specialist_sample_weights(frame, "voice")
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())


@pytest.mark.parametrize("profile", ["lp35", "lp52", "resample12_q12", "narrow_q8", "wide_q10"])
def test_codec_stress_profiles_are_deterministic(profile):
    wave = (0.1 * np.sin(np.linspace(0, 300, 32000))).astype(np.float32)
    first = apply_codec_sim(wave, profile=profile)
    second = apply_codec_sim(wave, profile=profile)
    assert first.shape == wave.shape
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


@pytest.mark.parametrize("profile", ["mulaw", "alaw", "narrow", "gsm_proxy"])
def test_telephone_stress_profiles_are_deterministic(profile):
    wave = (0.1 * np.sin(np.linspace(0, 300, 32000))).astype(np.float32)
    first = apply_telephone_sim(wave, profile=profile)
    second = apply_telephone_sim(wave, profile=profile)
    assert first.shape == wave.shape
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


@pytest.mark.parametrize("profile", ["narrow_q8", "wide_q10"])
def test_quantized_codec_stress_preserves_silence(profile):
    result = apply_codec_sim(np.zeros(32000, dtype=np.float32), profile=profile)
    assert np.array_equal(result, np.zeros_like(result))


def test_gsm_proxy_stress_preserves_silence():
    result = apply_telephone_sim(np.zeros(32000, dtype=np.float32), profile="gsm_proxy")
    assert np.array_equal(result, np.zeros_like(result))


def test_signed_training_quantizer_preserves_silence():
    result = signed_linear_quantize(np.zeros(32000, dtype=np.float32), bits=6)
    assert np.array_equal(result, np.zeros_like(result))


def test_voice_channel_v10_preserves_shape_range_and_finite_values():
    import random
    random.seed(31)
    np.random.seed(31)
    wave = (0.1 * np.sin(np.linspace(0, 300, 32000))).astype(np.float32)
    pipeline = AugmentationPipeline(profile="voice_channel_v10")
    for _ in range(20):
        result = pipeline(wave.copy())
        assert result.shape == wave.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert float(np.max(np.abs(result))) <= 1.0
