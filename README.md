# DACON 236749 audio deepfake detector

Offline submission pipeline for DACON competition 236749. The official score is
`0.9 * ADS + 0.1 * CPS`; ADS combines file/voice/music EER and CPS combines
voice/music presence AUC. Component EER is evaluated only where that component
is present. `src/metrics.py` mirrors the organizer's non-interpolated
`roc_curve(..., drop_intermediate=False)` implementation.

## Pipeline

- DF-Arena 1B INT8 ONNX: file fake evidence, exactly 64,600 samples, logits
  `[spoof, bonafide]` and class 0 mapped to fake.
- Independently trained voice/music SpecCNN specialists.
- Pretrained PANNs CNN14 AudioSet tags for continuous presence scores.
- One fusion function is shared by calibration and single/batch inference.
  Component fake scores are never multiplied by predicted presence.
- Adaptive auxiliary crops: 1 for <=8 s, 2 for <=25 s, 3 for longer audio.
  DF-Arena adds a distant second crop only for audio >=12 s whose first score
  is in the calibrated uncertainty interval.
- Rank-preserving output clipping only at `1e-6 .. 1-1e-6`.

HTDemucs is optional. If `--use_demucs` is requested, absence/load failure is
fatal; identity audio is never called a separated stem. Bundled checkpoints use
`use_demucs=False`, so default inference matches training. PANNs always receives
the original waveform.

## Data and validation

The 2,300 originals receive a stable `split_group_id` and are split before any mixing. Each split then independently
adds balanced RR/RF/FR/FF mixes in simultaneous, both sequential directions,
partial-overlap and crossfade layouts at multiple SNRs. Assertions reject base
recordings shared across train/validation.

- Train 1,799; VAL-A 328; VAL-B 291; VAL-C/D 328 each; independent fusion
  calibration 603; untouched final holdout 462.
- WF7 and AudioLDM2 fake generators are explicitly held out for VAL-B.
- VAL-C simulates codec/low-pass; VAL-D simulates telephone audio.
- Loss masks voice/music fake labels when the corresponding source is absent.
- Checkpoints use all four official validation scores, the worst split and
  unseen-generator VAL-B, not VAL-A alone.

## Reproduce

```powershell
python -m pytest -q
python -c "from src.dataset import scan_real_datasets,build_val_sets; build_val_sets(scan_real_datasets())"
python -m src.train --task voice --backbone spec_cnn --epochs 10 --batch_size 64 --device cpu --save_path model/best.pt
python -m src.train --task music --backbone spec_cnn --epochs 10 --batch_size 64 --device cpu --save_path model/music_best.pt
python scripts/calibrate_fusion.py --per_split 0 --batch_size 16 --device cpu
python script.py --test_dir data/test --output output/submission.csv
.\scripts\build_236749_submit.ps1
python tools/validate_submission.py submit.zip
```

The evaluator defaults are exactly `./data/test` and
`./output/submission.csv`. WAV, MP3, FLAC, M4A and OGG are accepted. All model
artifacts are mandatory and checkpoints load strictly/offline.

## Measured status (2026-08-31)

Ten CPU epochs were run per specialist. The selected checkpoints are voice
epoch 5 and music epoch 9. Three-fold robust calibration used all 603 samples.
The speed-selected voice-presence DF gate has robust objective 0.87466 versus
0.87623 for ungated primary-crop DF, while invoking DF for 64.0% of calibration
files. Its one-shot final-holdout score is 0.90487 (local data, not a DACON
leaderboard claim). On 64 local CPU files the same gate reduced DF calls from
64 to 41 and projected 1,200-file runtime from 9.68 to 7.45 minutes, a 23.0%
reduction. The official L4 benchmark and a new DACON score are **NOT RUN**.

Repository-authored code is MIT licensed. Third-party data/models retain their
upstream terms; see `docs/data_sources.md`.
