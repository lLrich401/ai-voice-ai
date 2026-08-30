import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.calibrate_fusion import validate_cache_metadata
from src.dataset import assert_no_base_source_overlap
from src.models.panns import PANNsPresenceWrapper
from src.train import specialist_sample_weights, validate_multisegment


def test_calibration_cache_rejects_every_stale_dependency():
    expected = {
        "voice_checkpoint_sha256": "voice", "music_checkpoint_sha256": "music",
        "df_model_sha256": "df", "panns_sha256": "panns",
        "split_csv_sha256": {"fusion_calibration.csv": "split"},
        "pipeline_version": "pipeline", "calibration_script_version": "calibration",
        "feature_extractor_version": "feature",
    }
    assert validate_cache_metadata(dict(expected), expected)
    for key in expected:
        stale = dict(expected)
        stale[key] = "changed"
        with pytest.raises(RuntimeError, match="Stale calibration cache"):
            validate_cache_metadata(stale, expected)


def test_saved_calibration_and_final_holdout_are_independent():
    root = pathlib.Path(__file__).resolve().parents[1] / "data/splits"
    train = pd.read_csv(root / "train.csv")
    model_selection = pd.concat([pd.read_csv(root / "val_a.csv"),
                                 pd.read_csv(root / "val_b.csv")], ignore_index=True)
    calibration = pd.read_csv(root / "fusion_calibration.csv")
    final_holdout = pd.read_csv(root / "final_holdout.csv")
    splits = {"train": train, "model_selection": model_selection,
              "fusion_calibration": calibration, "final_holdout": final_holdout}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            assert_no_base_source_overlap(splits[left], splits[right], (left, right))


def test_validate_multisegment_includes_mix_rows(monkeypatch):
    voice = np.linspace(-0.1, 0.1, 16000, dtype=np.float32)
    music = np.sin(np.linspace(0, 20, 16000)).astype(np.float32) * 0.1
    monkeypatch.setattr("src.dataset.load_audio",
                        lambda path, target_sr=16000: ((voice if "voice" in path else music), target_sr))
    rows = []
    for fake in (0, 1):
        rows.append({"path": "MIX::voice.wav|music.wav", "file_fake": fake,
                     "voice_fake": fake, "music_fake": fake,
                     "voice_present": 1, "music_present": 1, "augment": "none"})

    class ConstantModel(torch.nn.Module):
        def forward(self, wave):
            return {name: torch.zeros(len(wave), device=wave.device)
                    for name in ("file_fake", "voice_fake", "music_fake",
                                 "voice_present", "music_present")}

    metrics = validate_multisegment(ConstantModel(), pd.DataFrame(rows), torch.device("cpu"),
                                    use_demucs=False, task="voice", batch_size=2)
    assert metrics["score"] == pytest.approx(0.5)


def test_specialist_sampler_targets_component_mix_other_proportions():
    frame = pd.DataFrame({
        "path": ["voice.wav"] * 4 + ["MIX::v|m"] * 2 + ["music.wav"] * 4,
        "voice_present": [1] * 6 + [0] * 4,
        "music_present": [0] * 4 + [1] * 6,
    })
    weights = specialist_sample_weights(frame, "voice")
    assert weights[:4].sum() == pytest.approx(0.4)
    assert weights[4:6].sum() == pytest.approx(0.4)
    assert weights[6:].sum() == pytest.approx(0.2)


def test_actual_panns_checkpoint_has_near_complete_coverage():
    checkpoint = pathlib.Path(__file__).resolve().parents[1] / "model/panns/Cnn14_mAP=0.431.pth"
    if not checkpoint.exists():
        pytest.skip("bundled PANNs checkpoint unavailable")
    model = PANNsPresenceWrapper(use_pretrained=True)
    assert model.pretrained_loaded
    assert model.load_stats["key_coverage"] >= 0.98
    assert model.load_stats["element_coverage"] >= 0.98
    assert model.load_stats["missing_core"] == []
