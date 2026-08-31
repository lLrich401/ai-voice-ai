import pathlib
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.calibrate_fusion import validate_cache_metadata
from scripts.analyze_file_meta import assert_calibration_only, oof_logistic_scores
from scripts.audit_near_duplicates import spectral_fingerprint, fingerprint_similarity
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


def test_meta_fusion_is_deterministic_and_rejects_final_holdout():
    frame=pd.DataFrame({"calibration_fold":["a","a","b","b","c","c"],
                        "y_file_fake":[0,1,0,1,0,1],"split":["calibration"]*6})
    features=np.arange(18,dtype=float).reshape(6,3)
    first=oof_logistic_scores(frame,features,C=0.1,seed=7)
    second=oof_logistic_scores(frame,features,C=0.1,seed=7)
    np.testing.assert_allclose(first,second)
    frame.loc[0,"split"]="final_holdout"
    with pytest.raises(ValueError,match="final holdout"):
        assert_calibration_only(frame)


def test_audio_fingerprint_is_gain_tolerant_and_distinguishes_content():
    time=np.linspace(0,1,16000,endpoint=False)
    first=np.sin(2*np.pi*440*time).astype(np.float32)
    different=np.sin(2*np.pi*1700*time).astype(np.float32)
    fp_first,_=spectral_fingerprint(first)
    fp_gain,_=spectral_fingerprint(first*0.1)
    fp_different,_=spectral_fingerprint(different)
    assert fingerprint_similarity(fp_first,fp_gain)>0.999
    assert fingerprint_similarity(fp_first,fp_different)<0.95


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


def test_specialist_sampler_balances_sources_inside_bucket():
    frame=pd.DataFrame({"path":["a.wav"]*3+["b.wav"],"voice_present":[1]*4,
                        "music_present":[0]*4,"source":["large"]*3+["small"],
                        "generator":["real"]*4})
    weights=specialist_sample_weights(frame,"voice")
    assert weights[:3].sum()==pytest.approx(weights[3:].sum())


def test_actual_panns_checkpoint_has_near_complete_coverage():
    checkpoint = pathlib.Path(__file__).resolve().parents[1] / "model/panns/Cnn14_mAP=0.431.pth"
    if not checkpoint.exists():
        pytest.skip("bundled PANNs checkpoint unavailable")
    model = PANNsPresenceWrapper(use_pretrained=True)
    assert model.pretrained_loaded
    assert model.load_stats["key_coverage"] >= 0.98
    assert model.load_stats["element_coverage"] >= 0.98
    assert model.load_stats["missing_core"] == []


def test_runtime_source_hashes_match_root_source():
    root = pathlib.Path(__file__).resolve().parents[1]
    source = root / "src"
    runtime = root / "model/runtime/src"
    for path in source.rglob("*.py"):
        copied = runtime / path.relative_to(source)
        assert copied.exists(), f"runtime copy missing: {path.relative_to(source)}"
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(copied.read_bytes()).digest()
