import torch
import numpy as np

from src.models.music_forensic import (
    MusicForensicDualBranch, PANNsForensicHead, constant_q_filterbank,
)
from scripts.evaluate_v13b_file_complementarity import complementarity


def test_constant_q_filterbank_is_finite_normalized_and_log_spaced():
    filters = constant_q_filterbank()
    assert filters.shape == (72, 1025)
    assert torch.isfinite(filters).all()
    torch.testing.assert_close(filters.sum(dim=1), torch.ones(72))


def test_music_dual_branch_output_and_features():
    model = MusicForensicDualBranch(channels=4, embedding_dim=16).eval()
    wave = torch.randn(2, 64_000) * 0.01
    with torch.inference_mode():
        mel, cqt = model.features(wave)
        output = model(wave)
    assert mel.shape[:2] == (2, 96)
    assert cqt.shape[:2] == (2, 72)
    assert set(output) == {"music_fake", "file_fake"}
    assert all(value.shape == (2,) for value in output.values())
    assert all(torch.isfinite(value).all() for value in output.values())


def test_panns_forensic_head_is_small_and_frozen_backbone_independent():
    head = PANNsForensicHead().eval()
    with torch.inference_mode():
        output = head(torch.randn(3, 2048))
    assert set(output) == {"music_fake", "file_fake"}
    assert sum(parameter.numel() for parameter in head.parameters()) < 300_000


def test_file_error_complementarity_counts_every_row_once():
    truth = np.asarray([0, 0, 1, 1, 0, 1])
    first = np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
    second = np.asarray([0.2, 0.3, 0.8, 0.1, 0.9, 0.6])
    result = complementarity(truth, first, second)
    assert sum(result[key] for key in (
        "both_correct", "both_wrong", "only_canonical_wrong", "only_candidate_wrong")) == len(truth)
