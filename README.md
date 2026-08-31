# DACON 236749 audio deepfake detector

Offline, leakage-audited submission pipeline for DACON competition 236749.
The official score is `0.9 * ADS + 0.1 * CPS`; component fake EER is measured
only where the corresponding component exists. `src/metrics.py` implements the
organizer's non-interpolated ROC/EER calculation.

The machine-readable source of truth is
[`experiments/latest_results.json`](experiments/latest_results.json). Older
reports are historical and must not be quoted as current results.

## Selected pipeline

- DF-Arena 1B dynamic INT8 ONNX, 64,600 samples at 16 kHz, logits
  `[spoof, bonafide]`, class 0 = fake.
- Voice and music SpecCNN specialists, strictly loaded with checkpoint SHA
  validation. Voice fake segments use validation-selected `max` aggregation.
- Official PANNs `Cnn14_16k_mAP=0.438.pth`: 16 kHz, 512 FFT, 160 hop,
  64 mel bins, 50–8000 Hz, torchlibrosa frontend and active `bn0`.
- PANNs presence is blended at 0.75 with specialist presence. Component fake
  outputs are never multiplied by presence probabilities.
- DF gate is OFF and adaptive second crop is OFF. Every file receives one
  primary DF crop; this dominated the old voice-only gate on measured VAL-A/B
  and did not regress VAL-C/D.
- `use_demucs=False` in both checkpoint training and default inference.

## Data safety and validation

There are 2,785 original files grouped before augmentation by original
content, exact hash, decoded near-duplicate, speaker/source/generator and stable
`split_group_id`. Cross-family audit: 0 exact and 0 near duplicates.

- Train 2,025; VAL-A 533; VAL-B 293; fusion calibration 656; final holdout 463.
- VAL-B holds out unseen generators; VAL-C applies codec/compression changes;
  VAL-D applies telephone/narrow-band changes.
- Calibration uses three disjoint folds and `0.7 * mean + 0.3 * worst`.
- The v6 final holdout was evaluated once after policy selection. Its report
  script refuses a second execution.
- The manifest records license/provenance and exact content SHA. Only 500 rows
  are explicitly approved; 2,285 are `REVIEW_REQUIRED`. Current checkpoints
  retain those documented caveats. See `docs/data_sources.md`.

## Current measured result

| local final holdout | FILE EER | VOICE EER | MUSIC EER | ADS | VOICE AUC | MUSIC AUC | CPS | TOTAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v5 before | 0.11010 | 0.15163 | 0.01586 | 0.90986 | 0.90466 | 0.98639 | 0.94552 | 0.91343 |
| v6 after | 0.11010 | 0.16454 | 0.01586 | 0.90728 | 0.94143 | 0.98750 | 0.96446 | 0.91300 |

These are local measurements, not DACON leaderboard scores. The v6 total is
0.00043 lower on the one-shot local holdout, while CPS and non-final
cross-domain robustness improved. No post-holdout retuning was performed.

The selected batch size is 16. VAL-A 64-file throughput projects 12.79 minutes
for 1,200 files; the complete 463-file final run projects 14.75 minutes. Both
are local linear projections. Official server runtime is **NOT RUN**.

## Reproduce and package

```powershell
python -m pytest -q
python scripts/prepare_panns_16k.py  # one-time online artifact preparation
python scripts/audit_near_duplicates.py
python scripts/evaluate_panns_ab.py --splits fusion_calibration val_a val_b val_c val_d
python scripts/replace_cached_panns.py
python scripts/replace_cached_voice_aggregation.py
python scripts/calibrate_fusion.py --cache experiments/fusion_calibration_predictions_16k_voice_max.csv --reuse_cache --skip_final_holdout
python scripts/select_validated_policy.py
python scripts/benchmark_inference.py --samples 64 --split val_a --profiles selected_submission
.\scripts\build_236749_submit.ps1
python tools/validate_submission.py submit.zip
```

Do not rerun `scripts/evaluate_final_once.py`; its v6 one-shot report already
exists. The archive top level is exactly `model/`, `script.py`, and
`requirements.txt`. Training audio, unused checkpoints, `.ort` caches,
`__pycache__`, the legacy 32 kHz PANNs checkpoint and the upstream sampler
payload are excluded.

Repository-authored code is MIT licensed. Third-party data and models retain
their upstream terms.
