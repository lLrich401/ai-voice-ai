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
  validation. The v7 voice checkpoint was selected on fresh VAL-A/B/C/D
  reruns; voice segments retain `high_energy` selection and `max` aggregation.
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
  script refuses a second execution. **v7 did not run the final holdout.**
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

The v7 candidate was selected without touching that holdout. On fresh
canonical 128-sample VAL-A/B/C/D reruns, mean VOICE EER improved from
`0.171875` to `0.166667`; mean TOTAL improved from `0.906072` to `0.907062`;
and worst-domain TOTAL improved from `0.887188` to `0.887383`. Independent
calibration robust objective improved from `0.887414` to `0.888366`, with the
worst calibration fold improving from `0.880479` to `0.884531`. Detailed
per-domain results are in `experiments/v7/domain_results.json`.

`AFTER v7 FINAL HOLDOUT = NOT RUN`. The v6 final-holdout row above is
historical and must not be presented as a v7 score.

The submission default remains batch 16 because the model architecture is
unchanged and it was previously the stable L4-oriented choice. A post-training
local 64-file run measured 19.82 minutes projected at batch 8 and 22.18 minutes
at batch 16 after sustained CPU calibration load. These are noisy local linear
projections, not official server timings. Official server runtime is **NOT RUN**.

## v7 voice evidence

- Domain FP/FN, generator, speaker, source, duration, SNR, codec, telephone,
  mixed-music and confidence breakdowns are in
  `experiments/v7/voice_error_analysis.json` and
  `experiments/v7/voice_error_samples.csv`.
- Segment selection/aggregation, architecture, channel augmentation,
  partial-fake and TRAIN-only hard-mining ablations are under
  `experiments/v7/`. Rejected candidates are not bundled.
- New AASIST training and SSL comparison are explicitly **NOT RUN** because
  CUDA and a licensed offline SSL checkpoint were unavailable.
- Only 500 LibriSpeech rows are currently explicit `APPROVED`; that subset has
  no fake class, so approved-only training is **NOT RUN**, not silently treated
  as successful. See `experiments/v7/provenance_audit.json`.

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

## Project-created v8 data candidate

An independent, third-party-media-free procedural corpus is available through
`tools/generate_procedural_v8.py`. It uses only numeric seeds and repository
code: no external recording, lyrics, MIDI, pretrained generator, checkpoint,
or network service. The current deterministic build contains 930 synthetic
voice/music/mixed WAV files (160.4 MB), with 840 training rows and 90
content- and generator-disjoint stress rows. Its full audit is
`experiments/v8/generated_dataset_audit.json` and passes all quality, label,
provenance, exact-duplicate, near-duplicate, and split-isolation checks.

The corpus is deliberately **not selected into v7**. Follow-up risk testing
found an audio-only procedural-source classifier AUC of 1.0 for both voice and
music. Selected v7 also missed every procedural music fake at threshold 0.5.
The merge report therefore sets `training_authorized=false`; the files are
diagnostic-only until a more natural v8.2 generator passes the fingerprint and
VAL-A/B/C/D gates. Candidate manifests remain isolated under
`data/splits_v8_candidate/`, never over `data/splits/`. See
`docs/procedural_v8_dataset.md`.

Do not rerun `scripts/evaluate_final_once.py`; its v6 one-shot report already
exists. The archive top level is exactly `model/`, `script.py`, and
`requirements.txt`. Training audio, unused checkpoints, `.ort` caches,
candidate checkpoints, `__pycache__`, the legacy 32 kHz PANNs checkpoint and the upstream sampler
payload are excluded.

Repository-authored code is MIT licensed. Third-party data and models retain
their upstream terms.
