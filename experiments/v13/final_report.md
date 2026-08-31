# V13 radical redesign stage report

## CURRENT ACTUAL BASELINE

- Leaderboard ADS: `0.6386349206`
- Leaderboard CPS: `0.9366803175`
- Leaderboard TOTAL: `0.6684394603`
- Runtime: `30m52s`
- Source: user-reported TEST5 public leaderboard result

The existing TEST5 submission is frozen byte-for-byte in
`archive/pre_v13_selected/`. Its selected working-tree artifacts were not changed.

## DATASET V13

- real voice (train): `580`
- fake voice (train): `260`
- voice fake generator families (train): `11`
- voice unseen generator families (generator-disjoint validation): `4`
- real music (train): `240`
- fake music (train): `180`
- music fake generator families (train): `3`
- music unseen generator families (generator-disjoint validation): `1`
- mixed voice+music (train): `180`
- partial/crossfade (train): `0` — gap, not fabricated
- train: `1,080` approved rows
- CAL_V13: `550` approved rows
- generator-disjoint validation: `220` approved rows
- source-disjoint validation: `1,785` rows, all `REVIEW_REQUIRED`, not approved for training
- FINAL_HOLDOUT_V13: `795` rows, sealed
- manifest SHA256: `5e060f87fc854a568dcb4e6feddf312c10bf47b9aac6410adbe538fd1f6c072e`

Here, `approved` means the repository provenance manifest records competition use
as allowed; it is an engineering filter, not an independent legal opinion. Rows
marked `REVIEW_REQUIRED` are excluded from approved split files.

Final-holdout isolation audit found zero overlap with all development data for
source, split group, original ID, content group, near-duplicate group, audio SHA,
and fake generator family. It was then sealed and was not loaded for metrics,
model selection, or calibration; only its creation-time SHA and row count are exposed.

### Source-shortcut audit

| Probe | 5-fold AUC | Gate |
|---|---:|---:|
| Metadata only | 0.982362 | FAIL |
| Shallow acoustic only | 0.965769 | FAIL |
| Combined | 0.990098 | FAIL |

The pre-registered maximum was `0.75`. The pilot is therefore unsafe for forensic
model selection: source/channel identity nearly determines REAL/FAKE. In accordance
with the stage policy, model training and architecture comparison were stopped.

## PRE-V13 ERROR AUDIT

This audit used only VAL-A/B/C/D and expanded public unseen caches. It did not use
either old or V13 final holdout data.

| Scope | FILE EER | VOICE EER | MUSIC EER | FILE AUC | VOICE AUC | MUSIC AUC |
|---|---:|---:|---:|---:|---:|---:|
| VAL-A/B/C/D combined | 0.115104 | 0.171875 | 0.015625 | 0.932926 | 0.911323 | 0.998562 |
| Expanded unseen | 0.250000 | 0.025000 | 0.400000 | 0.829792 | 0.985000 | 0.733750 |

Measured failure concentrations include VAL-D voice EER `0.208333`, simultaneous
voice EER `0.360185`, partial-overlap voice EER `0.339080`, voice+music FILE EER
`0.15365`, and partial/crossfade FILE EER `0.134720`. Korean, explicit SNR/noise,
music vocality, and true fake occupancy were not consistently present in metadata
and are marked unavailable rather than inferred.

## MODEL RESULTS

### Required score comparison

| Model | FILE EER | VOICE EER | MUSIC EER | ADS | CPS | TOTAL |
|---|---:|---:|---:|---:|---:|---:|
| TEST5 public | NOT RUN | NOT RUN | NOT RUN | 0.6386349206 | 0.9366803175 | 0.6684394603 |
| TEST5 local VAL-A/B/C/D audit | 0.115104 | 0.171875 | 0.015625 | 0.903385 | NOT RUN | NOT RUN |
| V13 candidate | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

| Model | Voice unseen | Music unseen | FILE partial | Codec | Telephone | Worst |
|---|---:|---:|---:|---:|---:|---:|
| TEST5/V7 audit | 0.025000 | 0.400000 | 0.134720 | FILE 0.102083 / VOICE 0.166667 | FILE 0.147917 / VOICE 0.208333 | Music unseen 0.400000 |
| V13 candidate | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

| Model | Params | Size | Runtime 1200 |
|---|---:|---:|---:|
| TEST5 submission | NOT RUN | 892.292 MiB ZIP | NOT RUN |
| V13 candidate | NOT RUN | NOT RUN | NOT RUN |

| Model | ΔADS p05 | median | p95 | Bootstrap win |
|---|---:|---:|---:|---:|
| V13 candidate vs TEST5 | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

## BEST VOICE MODEL

- architecture: `NOT RUN`
- checkpoint: `NOT RUN`
- VAL EER: `NOT RUN`
- unseen EER: `NOT RUN`
- worst EER: `NOT RUN`

## BEST MUSIC MODEL

- architecture: `NOT RUN`
- checkpoint: `NOT RUN`
- VAL EER: `NOT RUN`
- unseen EER: `NOT RUN`
- worst EER: `NOT RUN`

## BEST FILE MODEL

- architecture: `NOT RUN`
- FILE EER: `NOT RUN`
- partial FILE EER: `NOT RUN`
- unseen FILE EER: `NOT RUN`

## FINAL LOCAL ROBUST RESULT

- FILE EER: `NOT RUN`
- VOICE EER: `NOT RUN`
- MUSIC EER: `NOT RUN`
- ADS: `NOT RUN`
- CPS: `NOT RUN`
- TOTAL: `NOT RUN`
- worst domain: `NOT RUN`
- voice unseen: `NOT RUN`
- music unseen: `NOT RUN`
- bootstrap p05 / median / p95 / win rate: `NOT RUN`

## RUNTIME

- 1200 projected: `NOT RUN`
- TEST5 official: `30m52s`
- current ZIP offline smoke: `PASS`, 3 generated 4-second files in 4.2 seconds
- model size: `892.292 MiB ZIP`

The three-file smoke timing is only a functional check and is not presented as a
1200-file runtime projection.

## FINAL HOLDOUT V13

- status: `SEALED — NOT RUN`
- FILE EER: `NOT RUN`
- VOICE EER: `NOT RUN`
- MUSIC EER: `NOT RUN`
- ADS: `NOT RUN`
- CPS: `NOT RUN`
- TOTAL: `NOT RUN`

## TESTS

- pytest: `114 passed`
- submission validator: `PASS`
- offline smoke: `PASS`
- archive top-level: exactly `model/`, `script.py`, `requirements.txt`

## FINAL DECISION

`KEEP_TEST5`

V13 is not adopted. Dataset Stage 3 failed before any architecture training, so
there is no defensible V13 checkpoint, fusion, bootstrap, or final-holdout result.

## SUBMISSION

- ZIP: `submit.zip` (unchanged TEST5 artifact)
- SHA256: `5898ad19b5c92e46d54aca529336b99f9e17838806766c654a3585051388fdcb`
- top-level: `model/`, `script.py`, `requirements.txt`
- offline smoke: `PASS`
- estimated/official runtime: `official TEST5 30m52s`; new estimate `NOT RUN`

## Required next data action

Do not add more one-class sources. Add content-matched REAL/FAKE pairs within the
same source and apply the same label-independent channel pipeline to both classes.
At minimum, the approved pilot still needs a second matched real/fake music domain,
approved source-disjoint development validation, and real partial-fake occupancy
examples. Re-run the shortcut audit before Stage 4; only an AUC at or below `0.75`
unblocks architecture training.
