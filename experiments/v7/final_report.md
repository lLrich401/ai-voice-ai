# v7 VOICE robustness report

All v7 selection used VAL-A/B/C/D plus the independent fusion-calibration
split. The already-consumed v6 final holdout was not loaded or evaluated.

## 1. 발견한 문제

- VOICE is the ADS bottleneck; VAL-D telephone remains the worst domain.
- VAL-D injected fresh random noise on every evaluation, making candidate
  comparisons nondeterministic. The transform is now deterministic.
- Historical validation caches did not contain enough checkpoint metadata to
  prove freshness; v7 decisions use fresh same-process checkpoint comparisons.
- The explicit approved-only subset contains only 500 LibriSpeech real rows and
  no fake class, so it cannot train/evaluate a valid detector.

## 2. 실제 수정한 것

- Added domain/sample VOICE FP/FN analysis, strict segment/architecture
  comparisons, TRAIN-only hard mining, controlled channel augmentation and
  partial-fake generation.
- Selected a fresh SpecCNN checkpoint only after all-domain validation and
  recalibrated fusion on 656 independent calibration rows.
- Locked DF always-on, adaptive DF OFF, PANNs 16 kHz and PANNs blend 0.75.
- Excluded all candidate checkpoints from the submission archive.

## 3. 수정하지 않은 것과 이유

- AASIST training: **NOT RUN**, CUDA unavailable.
- SSL backbone: **NOT RUN**, no licensed offline checkpoint bundled.
- Music/DF/PANNs architectures: unchanged; this iteration targeted VOICE and
  no evidence justified changing them.
- Final holdout: **NOT RUN**, already consumed by v6.

## 4. 데이터 변경

No external dataset was downloaded or silently approved. Partial-fake rows are
generated after the TRAIN split and were evaluated only as a rejected
experiment. Manifest counts: APPROVED 500, REVIEW_REQUIRED 2,285, REJECTED 0.

## 5. model 변경

The selected voice model remains SpecCNN but uses checkpoint SHA256
`ae354126b741f2212224da4ac6815558085ef892e32564baf2f5bb1cf326bac6`.

## 6. PANNs 변경

None: official 16 kHz CNN14, FFT 512, hop 160, 64 mel, 50–8000 Hz,
torchlibrosa, active bn0, strict SHA loading are retained.

## 7. DF gate 변경

None: gate OFF, one primary DF crop for every file, adaptive crop OFF.

## 8. fusion 변경

Independent calibration selected detector weights voice-file/music-file/OR =
0.0/0.5/0.5. DF file weight 0.25, DF voice-component 0.30 and PANNs presence
0.75 remain. Robust objective improved 0.887414 to 0.888366.

## 9. validation 구조

VAL-A general, VAL-B unseen generator, VAL-C codec, VAL-D telephone; independent
calibration folds cal_a/cal_b/cal_c. Final holdout was excluded.

## 10. leakage audit 결과

2,785 files; exact cross-family duplicates 0; near cross-family duplicates 0;
status PASS. Hard-mining input enforces `data_role=train` and exact TRAIN paths.

## 11. runtime

VAL-A 64-file local benchmark: batch 8 = 63.42 s, projected 1,200 files =
19.82 min; batch 16 = 70.99 s / 22.18 min. Official server runtime is NOT RUN.

## 12. submission size

`submit.zip`: 935,626,709 bytes, SHA256
`946e1e2d322c8e4d1df34ae2c7f8b74bd44fdb8c9f1b7269fba680a0ee861cb0`.
Archive layout and offline smoke inference PASS.

## 13. before / after metric

VOICE candidate comparison (fresh canonical validation):

| candidate | VAL-A | VAL-B | VAL-C | VAL-D | MEAN |
|---|---:|---:|---:|---:|---:|
| Baseline VOICE EER | 0.145833 | 0.166667 | 0.166667 | 0.208333 | 0.171875 |
| New VOICE EER | 0.145833 | 0.145833 | 0.166667 | 0.208333 | 0.166667 |

Final-holdout table:

| metric | BEFORE (v6 measured) | AFTER (v7) |
|---|---:|---:|
| FILE EER | 0.110101 | NOT RUN |
| VOICE EER | 0.164542 | NOT RUN |
| MUSIC EER | 0.015863 | NOT RUN |
| ADS | 0.907282 | NOT RUN |
| VOICE AUC | 0.941429 | NOT RUN |
| MUSIC AUC | 0.987497 | NOT RUN |
| CPS | 0.964463 | NOT RUN |
| TOTAL | 0.913000 | NOT RUN |
| worst domain | N/A | NOT RUN |
| runtime | 14.75 min historical projection | NOT RUN on final |

`AFTER FINAL HOLDOUT = NOT RUN`.

Validation-based selected candidate averages:

| metric | v7 validation |
|---|---:|
| FILE EER | 0.120833 |
| VOICE EER | 0.166667 |
| MUSIC EER | 0.020833 |
| ADS | 0.900000 |
| VOICE AUC | 0.945231 |
| MUSIC AUC | 0.996012 |
| CPS | 0.970622 |
| TOTAL | 0.907062 |
| worst-domain TOTAL | 0.887383 |

Calibration fold TOTAL: cal_a 0.884531, cal_b 0.897566, cal_c 0.887933;
worst fold cal_a. VAL-B 0.897331, VAL-C 0.915785, VAL-D 0.887383.
