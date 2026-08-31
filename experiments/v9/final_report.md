# v9 public-diversity result

All model selection used VAL-A/B/C/D and the independent fusion calibration
split. The already-consumed final holdout was not read or executed.

## Data

- 1,645 public candidate rows passed decoding and quality thresholds.
- New real music: GuitarSet, MTG-Jamendo, and exact FMA originals (three sources).
- New fake music: five SONICS and ten Echoes training generator families;
  DiffRhythm and SongGen are content-disjoint unseen validation generators.
- New fake voice: 12 MLAAD training TTS/VC families and four unseen validation
  families, all paired with their exact original content.
- Cross-role exact hash overlap: 0; cross-role split-group overlap: 0.
- Every Echoes/FMA content group contains a real original and generated audio.

## Voice candidate

| Candidate | VAL-A EER | VAL-B EER | VAL-C EER | VAL-D EER | Mean | Worst | Mean TOTAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected v7 SpecCNN | 0.14583 | 0.14583 | 0.16667 | 0.20833 | 0.16667 | 0.20833 | 0.90709 |
| v9 GPU AASIST | 0.14583 | 0.16667 | 0.14583 | 0.18750 | 0.16146 | 0.18750 | 0.90594 |

AASIST improved mean/worst EER but regressed the unseen-generator VAL-B and
mean TOTAL, so it failed the strict no-domain-regression gate and was rejected.

## Music candidate

Original validation:

| Candidate | VAL-A EER | VAL-B EER | VAL-C EER | VAL-D EER | Mean | Worst | Mean TOTAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected v7 SpecCNN | 0.02083 | 0.02083 | 0.02083 | 0.02083 | 0.02083 | 0.02083 | 0.90421 |
| v9 expanded SpecCNN | 0.02083 | 0.00000 | 0.02083 | 0.02083 | 0.01563 | 0.02083 | 0.89042 |

Expanded validation including content-disjoint Echoes/FMA:

| Candidate | VAL-A EER | VAL-B EER | VAL-C EER | VAL-D EER | Mean | Worst |
|---|---:|---:|---:|---:|---:|---:|
| selected v7 SpecCNN | 0.00000 | 0.18182 | 0.00000 | 0.00000 | 0.04545 | 0.18182 |
| v9 expanded SpecCNN | 0.01579 | 0.10245 | 0.01579 | 0.01317 | 0.03680 | 0.10245 |

The expanded candidate substantially reduced the new unseen-music worst domain,
but introduced small regressions in three established domains and its weaker
file head reduced original-domain TOTAL.

## Fusion and selection

| Calibration candidate | Robust objective |
|---|---:|
| selected v7 model and weights | 0.88837 |
| v9 music with current weights | 0.88300 |
| v9 music after coordinate recalibration | 0.88487 |

Recalibration did not recover the selected baseline. The selected v7 voice,
music, PANNs, DF-always-on policy, and fusion remain byte-for-byte unchanged.
`latest_results.json` was not updated because no candidate was adopted.

## Verification

- Public audio quality: 1,645/1,645 decoded, 0 threshold failures.
- Public cross-role exact duplicate/group leakage: PASS.
- Intel Arc B390 XPU AASIST training: measured, early-stopped at epoch 6;
  best epoch 3, final holdout NOT RUN.
- Intel Arc B390 XPU music SpecCNN training: measured 8 epochs; best epoch 6,
  final holdout NOT RUN.
- Candidate specialist inference on the fixed 512-row comparison: AASIST
  14.54 s; music SpecCNN 8.31 s.
- Test suite: 63 passed in 11.70 s.
- Submission archive rebuild: NOT RUN because both candidates were rejected.
