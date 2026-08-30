import numpy as np

import script


def test_df_arena_class_zero_is_fake():
    logits = np.array([[5.0, -5.0], [-5.0, 5.0]], dtype=np.float32)
    fake = script.df_arena_fake_probability(logits)
    assert fake[0] > 0.999
    assert fake[1] < 0.001
    assert script.DF_ARENA_LABELS == ("spoof", "bonafide")
    assert script.DF_ARENA_FAKE_INDEX == 0


def test_df_arena_preprocessing_uses_64600_and_tiles_short_audio():
    wave = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    segment = script._df_arena_segments(wave)[0]
    assert len(segment) == 64600
    np.testing.assert_array_equal(segment[:9], np.tile(wave, 3))
