import numpy as np
import pandas as pd

from src.dataset import (
    AudioDataset,
    add_split_internal_mixes,
    assert_disjoint_split_groups,
    assert_no_base_source_overlap,
    derive_split_group_id,
    mixed_labels,
    partial_fake_labels,
    render_partial_fake_wave,
    render_mixed_wave,
)


def _row(kind, fake, idx):
    voice = kind == "voice"
    return {
        "path": f"/{kind}/{fake}/{idx}.wav",
        "file_fake": fake,
        "voice_fake": fake if voice else 0,
        "music_fake": fake if not voice else 0,
        "voice_present": int(voice),
        "music_present": int(not voice),
        "speaker_id": f"{kind}_{idx}",
        "generator": f"{kind}_gen_{fake}",
        "source": kind,
        "dataset": kind,
        "hf_id": kind,
        "original_id": f"{kind}_{fake}_{idx}",
    }


def test_mixed_label_truth_table_and_balanced_classes():
    assert mixed_labels(0, 0) == [0, 0, 0, 1, 1]
    assert mixed_labels(0, 1) == [1, 0, 1, 1, 1]
    assert mixed_labels(1, 0) == [1, 1, 0, 1, 1]
    assert mixed_labels(1, 1) == [1, 1, 1, 1, 1]
    base = pd.DataFrame([_row(k, f, i) for k in ("voice", "music") for f in (0, 1) for i in range(2)])
    mixed = add_split_internal_mixes(base, mixes_per_class=3, random_state=7)
    generated = mixed[mixed["path"].str.startswith("MIX::")]
    counts = generated.groupby(["voice_fake", "music_fake"]).size().to_dict()
    assert counts == {(0, 0): 3, (0, 1): 3, (1, 0): 3, (1, 1): 3}
    assert set(generated["mix_mode"]) == {
        "simultaneous", "voice_then_music", "music_then_voice", "partial_overlap", "crossfade"
    }


def test_mix_modes_have_expected_lengths():
    voice = np.ones(16000, dtype=np.float32) * 0.1
    music = np.ones(8000, dtype=np.float32) * 0.1
    assert len(render_mixed_wave(voice, music, "simultaneous")) == 16000
    assert len(render_mixed_wave(voice, music, "voice_then_music")) == 24000
    assert len(render_mixed_wave(voice, music, "music_then_voice")) == 24000
    assert len(render_mixed_wave(voice, music, "partial_overlap")) == 20000
    assert len(render_mixed_wave(voice, music, "crossfade", crossfade_sec=0.25)) == 20000


def test_partial_fake_render_and_labels():
    real=np.zeros(1000,dtype=np.float32);fake=np.ones(100,dtype=np.float32)*0.5
    rendered=render_partial_fake_wave(real,fake,0.2,"middle",crossfade_sec=0.0,sr=1000)
    assert len(rendered)==len(real)
    assert np.count_nonzero(rendered)==200
    assert partial_fake_labels("voice")==[1,1,0,1,0]
    assert partial_fake_labels("music")==[1,0,1,0,1]


def test_base_source_leakage_detects_mixes():
    left = pd.DataFrame([_row("voice", 0, 0)])
    right = pd.DataFrame([_row("music", 0, 0)])
    assert_no_base_source_overlap(left, right)
    right.loc[0, "base_voice_id"] = left.loc[0, "original_id"]
    try:
        assert_no_base_source_overlap(left, right)
    except AssertionError:
        pass
    else:
        raise AssertionError("base-source leakage was not detected")


def test_mixed_specialists_receive_rendered_full_mix_without_demucs(monkeypatch):
    voice = np.linspace(-0.2, 0.2, 16000, dtype=np.float32)
    music = np.sin(np.linspace(0, 30, 16000)).astype(np.float32) * 0.1
    waves = {"voice.wav": voice, "music.wav": music}
    monkeypatch.setattr("src.dataset.load_audio", lambda path, target_sr=16000: (waves[path], target_sr))
    row = {
        "path": "MIX::voice.wav|music.wav", "file_fake": 1,
        "voice_fake": 0, "music_fake": 1, "voice_present": 1, "music_present": 1,
        "mix_mode": "simultaneous", "mix_snr_db": 0.0, "mix_crossfade_sec": 0.25,
    }
    frame = pd.DataFrame([row])
    expected = render_mixed_wave(voice, music)
    for task in ("voice", "music"):
        dataset = AudioDataset(frame, seg_sec=1.0, is_training=False, use_demucs=False, task=task)
        wave, labels, _ = dataset[0]
        np.testing.assert_allclose(wave.numpy(), expected, atol=1e-6)
        assert labels.tolist() == [1, 0, 1, 1, 1]
        assert not np.allclose(wave.numpy(), voice)
        assert not np.allclose(wave.numpy(), music)


def test_wavefake_generator_variants_share_split_group():
    first = {"source": "wavefake_ajay", "original_id": "wavefake_ajay_LJ019-0320_WF1"}
    second = {"source": "wavefake_ajay", "original_id": "wavefake_ajay_LJ019-0320_WF7"}
    assert derive_split_group_id(first) == derive_split_group_id(second) == "wavefake::LJ019-0320"
    left = pd.DataFrame([{**first, "path": "a.wav"}])
    right = pd.DataFrame([{**second, "path": "b.wav"}])
    try:
        assert_disjoint_split_groups({"train": left, "final_holdout": right})
    except AssertionError:
        pass
    else:
        raise AssertionError("WaveFake generator variants crossed major splits")


def test_four_major_split_groups_are_pairwise_disjoint():
    splits = {
        name: pd.DataFrame([{"path": f"{name}.wav", "source": "unit", "original_id": name}])
        for name in ("train", "model_selection", "fusion_calibration", "final_holdout")
    }
    counts = assert_disjoint_split_groups(splits)
    assert counts == {name: 1 for name in splits}
