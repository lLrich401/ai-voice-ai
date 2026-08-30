# Model report — 2026-08-31

## Executed work

- Official DACON metric parity and conditional component EER tests: PASS.
- Full rendered MIX input for both specialists when `use_demucs=False`: PASS.
- WaveFake LJ utterance grouping and four-family split leakage tests: PASS.
- Canonical MIX validation, separate component DF weights, calibrated DF gate,
  cache staleness, PANNs coverage and finite-output tests: PASS.
- `pytest -q`: 28 passed.
- Ten CPU epochs for each SpecCNN specialist: RUN.
- Official L4 and DACON leaderboard evaluation: NOT RUN.

## Disjoint data families

| Family | Rows | Originals | Mixes |
|---|---:|---:|---:|
| TRAIN | 1,799 | 1,399 | 400 |
| MODEL_SELECTION VAL-A | 328 | 248 | 80 |
| MODEL_SELECTION VAL-B | 291 | 211 | 80 |
| FUSION_CALIBRATION | 603 | 243 | 360 |
| FINAL_HOLDOUT | 462 | 302 | 160 |

Every pairwise base-recording overlap count between TRAIN,
MODEL_SELECTION, FUSION_CALIBRATION and FINAL_HOLDOUT is zero. Calibration
mixes contain RR/RF/FR/FF = 90/90/90/90 and final mixes contain
40/40/40/40.

## Checkpoints and calibration

| Specialist | Selected epoch (1-based) | Composite |
|---|---:|---:|
| Voice SpecCNN | 5 | 0.74932 |
| Music SpecCNN | 9 | 0.99782 |

The music checkpoint reaches zero local music EER on both original and mixed
holdout subsets. The source/generator breakdown also separates every current
GTZAN versus generated-music subgroup perfectly. This is evidence of remaining
source/domain shortcut risk, not proof of unseen-generator performance.

The selected primary-crop DF gate runs DF only when the voice specialist's
presence probability is at least 0.8. Robust CAL-A/B/C objective (`0.7 * mean +
0.3 * worst`) is **0.87466**, versus **0.87623** without the gate, using all
603 calibration rows. The gate invokes DF for 64.0% of calibration rows; fold
totals are 0.87597, 0.88321 and 0.87034.

## One-shot final holdout ablation

| Variant | File EER | Voice EER | Music EER | Voice AUC | Music AUC | ADS | CPS | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A previous models/fusion | 0.14264 | 0.17041 | 0.15165 | 0.91362 | 0.95326 | 0.84910 | 0.93344 | 0.85754 |
| B mixed-waveform specialists | 0.12556 | 0.14247 | 0.00758 | 0.90903 | 0.98471 | 0.90645 | 0.94687 | 0.91049 |
| C separate DF component weights | 0.14079 | 0.14247 | 0.00000 | 0.90914 | 0.98485 | 0.90111 | 0.94699 | 0.90570 |
| D calibrated adaptive DF | 0.14079 | 0.15365 | 0.00000 | 0.90914 | 0.98485 | 0.89888 | 0.94699 | 0.90369 |
| E speed-selected DF gate | 0.14264 | 0.14247 | 0.00000 | 0.90914 | 0.98485 | 0.90019 | 0.94699 | 0.90487 |

FINAL_HOLDOUT was not used to change the selected calibration weights.

## Runtime and package

On 64 local CPU files, ungated single-crop inference took 30.97 seconds and
projected to 9.68 minutes for 1,200 files. The selected gate took 23.85 seconds,
called DF for 41/64 files, and projected to 7.45 minutes: a 23.0% wall-time
reduction. On final holdout it called DF for 63.4% of rows. These are local CPU
projections, not official L4 measurements; decoding and evaluator overhead may
differ on the competition server.

The final archive is validated by fresh extraction and default offline
execution. Exact archive size and hash are recorded at build handoff.
