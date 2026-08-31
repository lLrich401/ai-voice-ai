# V12 final report

All numbers below are measured local validation results. The forbidden final holdout was not read or run.

## CALIBRATION BASELINE AUDIT

- old historical robust objective: `0.8883663604`
- fresh canonical robust objective: `0.8853084886`
- difference reason: the historical calibration script recorded `voice_fake_aggregation=max` but omitted the aggregation configuration while collecting features, so it actually used the default top-2 mean. Fresh canonical extraction correctly uses max. The current DF-Arena ONNX serialization SHA also differs from historical metadata, but this did not cause the old-vs-V11 delta because both old runs reused the same cached DF predictions.

## CAL_V12

- rows: `500` (RR/RF/FR/FF: 125 each)
- folds: 167 / 167 / 166
- voice generator/source families: `12`
- music generator/source families: `12`
- real sources: `4`
- train overlap: speaker/original/split-group/base-audio/near-duplicate all `0`
- validation and expanded-unseen overlap: all audited keys `0`
- final holdout: `NOT READ / NOT RUN`

CAL_V12 intentionally stresses fake-component generalization. All rows contain both components, so presence AUC is not identifiable there and resolves to 0.5; CPS selection still relies on VAL-A/B/C/D and CAL_OLD. This limitation is not hidden as a measured CPS result.

## MUSIC candidates

| Candidate | FILE EER | MUSIC EER | Music unseen | ADS | TOTAL | CAL OLD | CAL V12 | Robust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 | 0.120833 | 0.020833 | 0.4000 | 0.900000 | 0.907058 | 0.885308 | 0.588535 | 0.760941 |
| M4 | 0.123437 | 0.020833 | 0.4000 | 0.898698 | 0.905903 | 0.886047 | 0.591988 | 0.760149 |

Best music point candidate: `model/candidates/v12/m4_student.pt`, SHA256 `033030fb1511a33bef89e4ab807a0752b60d50131102d2bf17f55da601af16c8`.

Training configuration: 0.60 supervised + 0.25 V7 retention + 0.15 V9 MUSIC_FAKE distillation, temperature 2, EMA enabled, LR 3e-5, V7 file-head retention. It was rejected: robust point estimate decreased and paired-bootstrap robust win rate was only 0.524.

## VOICE candidates

| Candidate | FILE EER | VOICE EER | Voice unseen | ADS | TOTAL | CAL OLD | CAL V12 | Robust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 | 0.120833 | 0.166667 | 0.4000 | 0.900000 | 0.907058 | 0.885308 | 0.588535 | 0.760941 |
| V2 | 0.120833 | 0.166667 | 0.3250 | 0.900000 | 0.907213 | 0.885759 | 0.586882 | 0.772556 |

Best voice point candidate: `model/candidates/v12/v2_student.pt`, SHA256 `378380067ce9a751c301ca51a446e4ae8f9a8314e20f4af643f2bd4d6b890b23`.

Training configuration: 0.80 supervised + 0.20 V7 retention, LR 3e-5, file-head LR multiplier 0.1. V2 is the strongest robust voice candidate, but it is not the AASIST-distilled candidate; the AASIST-distilled V3/V4/V5 variants were evaluated and did not beat V2 under the precommitted objective. V2 was not adopted alone because CAL_V12 decreased and its bootstrap ADS median was negative.

## JOINT

| Candidate | FILE EER | VOICE EER | MUSIC EER | ADS | CPS | TOTAL | Worst | Voice unseen | Music unseen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V7 | 0.120833 | 0.166667 | 0.020833 | 0.900000 | 0.970581 | 0.907058 | 0.887334 | 0.4000 | 0.4000 |
| M4 + V2 | 0.125000 | 0.166667 | 0.020833 | 0.897917 | 0.972290 | 0.905354 | 0.883161 | 0.3250 | 0.4000 |

Joint calibration: CAL_OLD `0.886498`, CAL_V12 `0.590336`, robust objective `0.771588`. Conservative fusion fit on BOTH raised `w_df_voice_component` from 0.30 to 0.35 but reduced validation mean/worst to 0.904182/0.879411, so it was rejected. CAL_V12-only fitting raised `w_df_arena` to 0.50, improved CAL_V12 to 0.593266, but reduced CAL_OLD to 0.870798 and was also rejected. Logistic FILE meta fusion was not run because the prerequisite of no FILE EER regression failed.

## BOOTSTRAP (1,000 paired resamples)

| Candidate | Robust win | ΔADS p05 | ΔADS median | ΔADS p95 | Δrobust median |
|---|---:|---:|---:|---:|---:|
| M4 | 0.524 | -0.007572 | -0.001645 | 0.003499 | 0.000097 |
| V2 | 0.942 | -0.006342 | -0.001047 | 0.003009 | 0.009913 |
| M4 + V2 | 0.907 | -0.010230 | -0.003063 | 0.001531 | 0.009726 |

The joint robust signal is largely driven by expanded-unseen voice improvement. Its ADS and TOTAL bootstrap medians are negative, and FILE EER regresses; this is not safe enough for submission adoption.

## RUNTIME

| Candidate | Projected 1200 files | Model count |
|---|---:|---:|
| V7 | 19.8178 min | 4 |
| M4 + V2 student-only | 19.8172 min | 4 |

Measured locally on XPU using 64 files and three specialist timing repetitions, combined with the existing full-pipeline projection. Official evaluator runtime: `NOT RUN`.

## TESTS

`pytest -q`: **100 passed in 8.72s**.

## FINAL DECISION

**KEEP_V7**.

The joint V12 candidate passes the robust bootstrap threshold and improves CAL_OLD/CAL_V12 and voice unseen EER, but fails the required FILE preservation and stable ADS conditions. The selected V7 checkpoint, music checkpoint, fusion JSON, and submission script remain byte-identical to their frozen SHA256 values.

## SUBMISSION

- built: `BUILT FROM UNCHANGED SELECTED V7` after an explicit user request
- validator: `PASS` (archive layout, mandatory artifacts, strict model loading, offline smoke inference)
- zip SHA256: `5898ad19b5c92e46d54aca529336b99f9e17838806766c654a3585051388fdcb`
- final holdout: `NOT RUN`
