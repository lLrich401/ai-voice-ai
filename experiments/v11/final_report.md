# V11 head-selective ensemble report

Status: **KEEP_V7**
Final holdout: **NOT RUN**
Official DACON runtime: **NOT RUN**

All figures below are measured on VAL-A/B/C/D, independent fusion-calibration folds,
or the expanded unseen sets. They are not leaderboard or final-holdout results.

## Integrity and implementation

- The selected v7 checkpoints, fusion config, and `script.py` were frozen by SHA256.
- All v7 and v9 prediction heads were recomputed on the same rows and segments. A stale
  historical v7 cache whose aggregation no longer matched the selected pipeline was discarded.
- Every cache records checkpoint SHA256, split SHA256, pipeline version, and aggregation version.
- Candidate FILE heads are disabled in the component-only experiments.
- Probability, logit, rank, and max blending were searched. Rank blending was measured but
  excluded from deployment because a file's rank depends on the other files in the batch.
- `FINAL_HOLDOUT_FORBIDDEN=1` guards all v11 evaluation paths.

## Candidate comparison — VAL-A/B/C/D mean

| Candidate | FILE EER | VOICE EER | MUSIC EER | ADS | CPS | TOTAL | Worst TOTAL | Unseen TOTAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — v7 | 0.120833 | 0.166667 | 0.020833 | 0.900000 | 0.970622 | 0.907062 | 0.887383 | 0.897331 |
| B — music only | 0.120833 | 0.166667 | 0.015625 | 0.901563 | 0.970622 | 0.908468 | 0.887383 | 0.902956 |
| C — voice only | 0.120833 | 0.156250 | 0.020833 | 0.902083 | 0.970622 | 0.908937 | 0.893581 | 0.893581 |
| D — joint | 0.120833 | 0.156250 | 0.015625 | 0.903646 | 0.970622 | 0.910343 | 0.894883 | 0.899206 |
| E — D + constrained fusion | 0.110938 | 0.161458 | 0.020833 | 0.905990 | 0.970622 | 0.912453 | 0.891133 | 0.903893 |
| F — logistic FILE meta | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

`Unseen TOTAL` is the configured VAL-B unseen-generator selection signal. Expanded unseen
component EERs are shown separately below.

## Domain and unseen comparison

| Candidate | Voice unseen EER | Music unseen EER | VAL-C TOTAL | VAL-D TOTAL |
|---|---:|---:|---:|---:|
| A — v7 | 0.400000 | 0.400000 | 0.915785 | 0.887383 |
| B — music only | 0.400000 | 0.312500 | 0.915785 | 0.887383 |
| C — voice only | 0.275000 | 0.400000 | 0.919535 | 0.894883 |
| D — joint | 0.275000 | 0.312500 | 0.919535 | 0.894883 |
| E — D + constrained fusion | 0.275000 | 0.312500 | 0.927035 | 0.891133 |

## Selected head blends

- B: music alpha `1.00`, probability blend, component-only.
- C: voice alpha `0.75`, probability blend, component-only.
- D: voice alpha `0.75` + music alpha `1.00`, probability blends, component-only.
- E: D plus `w_df_voice_component=0.35`, `w_df_music_component=0.05`,
  `w_df_arena=0.25`; candidate FILE heads remain disabled.

## Paired bootstrap on VAL-A/B/C/D — 1000 iterations

The EER delta fields in the raw JSON use quality direction for specialist EERs. The ADS fields
below are reported directly as candidate minus v7.

| Candidate | Robust win rate | ΔADS p05 | ΔADS median | ΔADS p95 |
|---|---:|---:|---:|---:|
| B — music only | 0.461 | -0.006857 | -0.001355 | 0.003085 |
| C — voice only | 0.443 | -0.005081 | 0.000132 | 0.005106 |
| D — joint | 0.484 | -0.008099 | -0.001118 | 0.005276 |
| E — constrained fusion | 0.483 | NOT RUN | NOT RUN | NOT RUN |

For E, the exact-candidate robust delta was: p05 `-0.008541`, median `-0.000170`,
p95 `0.009312`. Its validation and calibration point-estimate robust deltas were respectively
`+0.005335` and `+0.002297`, but the bootstrap evidence was not stable enough for adoption.

## Independent fusion-calibration check

| Candidate | Robust objective | Delta vs v7 | Bootstrap robust win rate | Adopt gate |
|---|---:|---:|---:|---:|
| A — v7 | 0.885310 | 0.000000 | — | baseline |
| B — music only | 0.883623 | -0.001687 | 0.127 | fail |
| C — voice only | 0.885221 | -0.000089 | 0.405 | fail |
| D — joint | 0.883534 | -0.001776 | 0.215 | fail |

Candidate F was not run because candidate E failed the stable-improvement gate.

## Runtime

Measured locally on CPU with 64 VAL-A files, three repetitions, shared decoding and segment
extraction. The 1200-file projections add measured specialist overhead to the existing v7
full-pipeline projection; they are not official server measurements.

| Pipeline | Projected 1200-file runtime | Under 60 min | Under preferred 35 min |
|---|---:|:---:|:---:|
| v7 baseline | 19.82 min | yes | yes |
| + v9 music | 19.91 min | yes | yes |
| + v9 AASIST voice | 36.39 min | yes | no |
| joint | 36.48 min | yes | no |

## Final decision

**KEEP_V7.** The head-selective candidates improve the point estimates and expanded unseen
EERs, but none passed the independent calibration and paired-bootstrap stability requirements.
The v7 selected artifacts therefore remain byte-for-byte selected, and candidate checkpoints
are not included in `submit.zip`.

Selected SHA256:

- `model/best.pt`: `ae354126b741f2212224da4ac6815558085ef892e32564baf2f5bb1cf326bac6`
- `model/music_best.pt`: `ed87097507ed89991dd49952fbdcb9c5ceb0c256d871a385d9cdbcf9945c84c1`
- `model/fusion_weights.json`: `87d46c317398bed0a9dc87c6b451246851ba6c34683e4468a0251461f7c42402`
- `script.py`: `8abd4888f09aa00be18790e5257bf0cafc2fff5fd9e167571c880509a665119b`
