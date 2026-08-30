import torch

from src.train import checkpoint_selection_score, masked_multitask_loss


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
