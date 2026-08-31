# DACON 236749 final improvement report

Generated: 2026-08-31. All values below are measured results unless explicitly marked as a projection or NOT RUN.

## Starting point

- User-reported public submission: TOTAL 0.6461396931, ADS 0.6276746032, CPS 0.8123255026, runtime 15m 50s.
- The old local holdout was invalid for music comparison: 500 GTZAN manifest rows represented only 10 unique file hashes and leaked across split families.

## Data and leakage repair

- Recovered the official cached GTZAN parquet snapshot: 999 source rows.
- Removed 14 exact source duplicates, leaving 985 unique file hashes.
- Grouped 13 decoded near-duplicate pairs (26 rows) using a label-free spectro-temporal fingerprint plus waveform correlation >= 0.999.
- Regenerated every split by original-content group.
- Cross-family audit after regeneration: 2,785 files checked, 0 exact and 0 near duplicates (`PASS`).
- Final split sizes: train 2,025; VAL-A 533; VAL-B 293; fusion calibration 656; final holdout 463.

## Selected models and calibration

- Voice SpecCNN: early-stopped after epoch 6; epoch 1 checkpoint retained by the four-condition composite objective.
- Music SpecCNN: epoch 10 selected; composite 0.9937.
- Music multi-segment validation EER: VAL-A 0.000, VAL-B 0.000, codec VAL-C 0.000, telephone VAL-D 0.000.
- Fusion used three disjoint calibration folds only. Robust calibration objective: 0.8877316689.
- Selected FILE weights: voice 0.25, music 0.50, probability-OR 0.25, DF-Arena 0.25.
- DF component weights: voice 0.30, music 0.00. Voice-presence gate threshold: 0.80.
- Calibration DF gate fraction: 0.6219512195.
- Selected adaptive policy: primary uncertainty 0.20-0.80, max aggregation, minimum duration 12s.

## Untouched final holdout

| Variant | FILE EER | VOICE EER | MUSIC EER | ADS | CPS | TOTAL |
|---|---:|---:|---:|---:|---:|---:|
| Existing Git baseline on repaired split | 0.17278 | 0.19351 | 0.14379 | 0.83177 | 0.93792 | 0.84238 |
| Retrained specialists | 0.15979 | 0.16454 | 0.03173 | 0.87768 | 0.94552 | 0.88446 |
| Calibrated FILE/DF weights | 0.11010 | 0.16454 | 0.01586 | 0.90728 | 0.94552 | 0.91111 |
| Adaptive DF crop | 0.11010 | 0.15809 | 0.01586 | 0.90857 | 0.94552 | 0.91227 |
| Selected submission | 0.11010 | 0.15163 | 0.01586 | 0.90986 | 0.94552 | 0.91343 |

The selected submission improves the repaired-split baseline TOTAL by 0.07105. These local values are not a guarantee of the private leaderboard score.

## Runtime

Measured on 32 real files, CPU, batch size 16, after model warm-up:

| Profile | DF calls | Seconds | Projected 1,200-file time |
|---|---:|---:|---:|
| Selected submission | 24 | 20.2306 | 12.6441 min |
| Gate, no adaptive crop | 19 | 16.0774 | 10.0483 min |
| No DF-Arena | 0 | 1.7828 | 1.1143 min |

The earlier accuracy profile projected 20.2638 minutes locally; the selected submission projection is 37.6% lower. The new official evaluator runtime is **NOT RUN**; the 12.64-minute value is a local linear projection, not an official measurement.

## Rejected or unavailable experiments

- FILE OOF logistic meta-fusion: rejected. Bootstrap win probability 0.545 and improvement p05 -0.00690 failed the adoption rule.
- Lower DF gates: rejected when they reduced the worst calibration fold.
- Source-level music domain holdout: NOT RUN because only one real-music source and two fake generators were legally available.
- AASIST three-seed GPU comparison and music multi-seed training: NOT RUN because this machine has no CUDA. Reproducible runners and status files are included.
- Partial-fake training: infrastructure and tests added, but no checkpoint was claimed without a completed controlled retraining run.

## Verification and artifact

- `python -m pytest -q`: 37 passed.
- `python tools/validate_submission.py submit.zip`: PASS, including a three-file end-to-end inference smoke test.
- `submit.zip`: 928,328,170 bytes.
- Package contains `script.py`, requirements, two selected SpecCNN checkpoints, DF-Arena INT8 ONNX, PANNs, calibrated weights, and a byte-identical runtime source copy.
