import numpy as np
import pandas as pd

from src.dataset import (
    add_split_internal_mixes,
    assert_no_base_source_overlap,
    mixed_labels,
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
    assert len(render_mixed_wave(voice, music, "partial_overlap")) == 16000
    assert len(render_mixed_wave(voice, music, "crossfade", crossfade_sec=0.25)) == 20000


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
