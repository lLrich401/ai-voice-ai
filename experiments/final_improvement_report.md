# DACON 236749 v6 measured report

Generated 2026-08-31. `latest_results.json` is authoritative. Every value is a
local measurement unless marked user-reported, projected, or NOT RUN.
Implementation commit: `ef828d95401f110cbdfdeb4ee77effa38865b054`.

## 1. Problems found and changes made

- The 32 kHz `Cnn14_mAP=0.431.pth` was fed a custom 16 kHz 1024/320 frontend;
  torchaudio dB conversion differed from upstream and `bn0` was skipped.
- Replaced it with official 16 kHz CNN14 (512/160/64 mel/50–8000 Hz), exact
  torchlibrosa flow and strict 84/84 state loading. The packaged state dict
  removes only the upstream training sampler.
- Calibration PANNs CPS changed from 0.46993 (mismatched) to 0.86890 standalone;
  0.75 specialist blending measured 0.94378.
- Rejected the old voice-presence 0.8 DF gate. Gate OFF + one primary crop
  improved balanced VAL-A/B by 0.00375 and tied VAL-C/D.
- Rejected adaptive second crop despite a higher four-domain mean because
  codec VAL-C regressed by 0.00375.
- Voice `max` segment aggregation lowered raw VOICE EER on all four validation
  domains and caused no fused TOTAL regression; it replaced `topk_mean`.
- Vectorized fusion scoring and replaced the 5-D Cartesian calibration grid
  with two-pass coordinate search, reducing a stalled minutes-long search to
  about nine seconds and reducing calibration freedom.
- Removed duplicate separator construction, fixed runtime benchmark gate
  emulation, prohibited final-holdout runtime tuning, and kept batch size 16.
- Added manifest-level provenance, exact content SHA, explicit review status,
  opt-in approved-only training, and fail-closed path-label inference.

## 2. Data, validation and leakage

- No new training audio was downloaded and no DACON test audio was used.
- 2,785 originals; exact cross-family duplicates 0; decoded near duplicates 0.
- Independent train/model-selection/calibration/final families remain intact.
- Music source-disjoint experiment remains NOT RUN: one real source and two
  fake generators are insufficient. The very low local MUSIC EER is therefore
  not treated as proof of private generalization.
- 2,285 provenance rows require license review; only LibriSpeech's 500 rows are
  explicitly marked approved. Current checkpoints were not retrained under the
  strict approved-only filter.

## 3. Before / after

| metric | BEFORE v5 | AFTER v6 |
|---|---:|---:|
| FILE EER | 0.110101 | 0.110101 |
| VOICE EER | 0.151631 | 0.164542 |
| MUSIC EER | 0.015863 | 0.015863 |
| ADS | 0.909864 | 0.907282 |
| VOICE AUC | 0.904659 | 0.941429 |
| MUSIC AUC | 0.986390 | 0.987497 |
| CPS | 0.945525 | 0.964463 |
| TOTAL | 0.913430 | 0.913000 |
| projected local runtime / 1,200 | 12.64 min | 12.79–14.75 min |
| DF primary fraction | 0.621951 | 1.000000 |

The final candidate was locked before the v6 one-shot holdout. Its TOTAL is
0.00043 lower there, but reverting would tune on the final holdout and violate
the selection protocol. Non-final domain robustness was the selection target.

## 4. Calibration and domain results

Final gate-off calibration robust objective: 0.887414. Fold totals: CAL-A
0.880479, CAL-B 0.898259, CAL-C 0.892418; worst fold CAL-A.

Balanced 128-row non-final measurements with final voice aggregation:

| condition | TOTAL |
|---|---:|
| VAL-A | 0.926725 |
| VAL-B unseen generator | 0.898021 |
| VAL-C codec | 0.915199 |
| VAL-D telephone | 0.887171 |

## 5. Runtime and submission

- Batch 8/16/24/32 projected 14.18/12.79/13.82/14.45 minutes from VAL-A;
  batch 16 retained.
- Complete v6 holdout: 463 files in 341.50 seconds, linear projection 14.75 min.
- Official DACON runtime: NOT RUN.
- `submit.zip`: 935,623,738 bytes; top level exactly `model/`, `script.py`,
  `requirements.txt`.
- Archive validator and offline three-file end-to-end smoke test: PASS.
- `pytest`: 38 passed.

## 6. Not changed

- No new AASIST or partial-fake checkpoint was claimed because controlled GPU
  retraining was not completed.
- No meta-fusion was adopted; prior OOF bootstrap evidence remained weak.
- No private/public leaderboard feedback was used for hyperparameter search.
- Component outputs remain ungated by presence, and output post-processing is
  limited to finite probability clipping.
