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
  DF-Arena keeps one energy crop to meet the evaluation time limit.
- Rank-preserving output clipping only at `1e-6 .. 1-1e-6`.

HTDemucs is optional. If `--use_demucs` is requested, absence/load failure is
fatal; identity audio is never called a separated stem. Bundled checkpoints use
`use_demucs=False`, so default inference matches training. PANNs always receives
the original waveform.

## Data and validation

The 2,300 originals are split before any mixing. Each split then independently
adds balanced RR/RF/FR/FF mixes in simultaneous, both sequential directions,
partial-overlap and crossfade layouts at multiple SNRs. Assertions reject base
recordings shared across train/validation.

- Train 1,786; VAL-A 424; VAL-B 539; VAL-C/D 424 each.
- WF7 and AudioLDM2 fake generators are explicitly held out for VAL-B.
- VAL-C simulates codec/low-pass; VAL-D simulates telephone audio.
- Loss masks voice/music fake labels when the corresponding source is absent.
- Checkpoints use all four official validation scores, the worst split and
  unseen-generator VAL-B, not VAL-A alone.

## Reproduce

```powershell
python -m pytest -q
python -c "from src.dataset import scan_real_datasets,build_val_sets; build_val_sets(scan_real_datasets())"
python -m src.train --task voice --backbone spec_cnn --epochs 5 --batch_size 32 --device cpu --save_path model/best.pt
python -m src.train --task music --backbone spec_cnn --epochs 5 --batch_size 32 --device cpu --save_path model/music_best.pt
python scripts/calibrate_fusion.py --per_split 60 --batch_size 12 --device cpu
python script.py --test_dir data/test --output output/submission.csv
.\scripts\build_236749_submit.ps1
python tools/validate_submission.py submit.zip
```

The evaluator defaults are exactly `./data/test` and
`./output/submission.csv`. WAV, MP3, FLAC, M4A and OGG are accepted. All model
artifacts are mandatory and checkpoints load strictly/offline.

## Measured status (2026-08-30)

Five CPU epochs were run per specialist. Selected responsibility scores are
0.81388 for voice (epoch 5) and 1.0 for music (epoch 1). See
`docs/model_report.md`. A 20-file repeated-input steady-state CPU benchmark took
10.027 s, or a linear 1,200-file projection of 10.027 min (not an official L4
measurement). The owner's earlier DACON result—total 0.46623, ADS
0.42838, CPS 0.80685—predates this overhaul. A new DACON score is **NOT RUN**.

Repository-authored code is MIT licensed. Third-party data/models retain their
upstream terms; see `docs/data_sources.md`.
