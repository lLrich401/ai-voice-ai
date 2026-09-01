# V13B Representation Research Report

## CURRENT SELECTED

- Selected submission: `TEST5` (unchanged)
- ADS: `0.6386349206` (`USER_REPORTED_PUBLIC`)
- CPS: `0.9366803175` (`USER_REPORTED_PUBLIC`)
- TOTAL: `0.6684394603` (`USER_REPORTED_PUBLIC`)
- Runtime: `30m52s` (`USER_REPORTED_PUBLIC`)
- Frozen ZIP SHA256: `5898ad19b5c92e46d54aca529336b99f9e17838806766c654a3585051388fdcb`

## CURRENT COMMIT

- Branch: `x1`
- development_base_commit: `e8434b9c368ee5de3368d8b0b04559cf19c3ffaa`
- current_git_commit: `a593422aa342c6579f8e3cc799c65e0dad73ad32`

## DATA

- Train: `546` rows; calibration: `112`; generator-disjoint validation: `132`
- Paired voice sources: `2`; paired Music sources: `1`
- Second approved paired Music source: `NOT ACQUIRED`
- Metric-complete source-disjoint validation: `NOT ACQUIRED / NOT MEASURED`
- Globally unused final: `NOT ACQUIRED / NOT SEALED / NOT RUN`
- MuseBench, HAIM and ArtifactBench were rechecked from their official dataset cards. Heterogeneous per-track rights, NC terms, URL-only real arms, or historical-source overlap remain; no row was auto-approved or downloaded.

## SHORTCUT EFFECTIVE AUC

The symmetric definition is `effective AUC = max(raw AUC, 1 - raw AUC)` and `distance = abs(raw AUC - 0.5)`.

| Audit | Raw AUC | Effective AUC | Distance from random |
|---|---:|---:|---:|
| Direct metadata | 0.581653 | 0.581653 | 0.081653 |
| Direct acoustic | 0.641359 | 0.641359 | 0.141359 |
| Direct combined | 0.718194 | 0.718194 | 0.218194 |
| Rendered all | 0.551398 | 0.551398 | 0.051398 |
| Rendered paired | 0.635346 | 0.635346 | 0.135346 |
| Rendered partial | 0.403320 | 0.596680 | 0.096680 |
| Rendered mixed | 0.538194 | 0.538194 | 0.038194 |

Decision: `PASS <= 0.75`. The earlier interpretation of partial raw AUC `0.4033` as inherently better than random was corrected.

## MUSIC ARCHITECTURE TABLE

| Candidate | Representation | Generator-disjoint Music EER | Source-disjoint Music EER | Delta ADS Music | Runtime |
|---|---|---:|---|---:|---|
| M0 | TEST5 Music SpecCNN | 0.312500 | NOT MEASURED | 0.000000 | 1.99 s / 132 rows, specialist-only MEASURED |
| M1 | Log-mel + STFT constant-Q dual branch | 0.375000 | NOT MEASURED | -0.018750 | train 122.07 s; eval 8.73 s MEASURED |
| M2 | Official PANNs 16 kHz frozen embedding + small head | 0.312500 | NOT MEASURED | 0.000000 | train/embedding 59.60 s; eval 46.56 s MEASURED |
| M3 | ArtifactNet v9.4 forensic residual ONNX | 0.125000 median / 0.187500 one-crop, DIAGNOSTIC | NOT MEASURED | +0.056250 median / +0.037500 one-crop, DIAGNOSTIC | +54.31 min median / +18.29 min one-crop if all 1200, PROJECTED |

- M1: `REJECT_REGRESSION`
- M2 Music: `REJECT_NO_IMPROVEMENT`
- Neither reached the clear screening threshold `EER <= 0.28`; no partial unfreeze or full fine-tune was run.
- The failed SpecCNN, M1 and M2 Music evidence is preserved under `experiments/v13b/rejected/`.
- M3 showed a strong diagnostic representation signal, but failed production screening: 1/95 segments produced NaN and was explicitly skipped only in diagnostic mode. The default evaluator fails closed. Its source-disjoint score is unavailable, CC-BY-NC/patent competition approval is unconfirmed, and its CPU runtime is outside the all-file budget. It was not copied into selected artifacts or `submit.zip`.

## FILE ARCHITECTURE TABLE

| Candidate | FILE EER | Partial FILE EER | Error correlation vs DF/Fusion | Estimated runtime |
|---|---:|---:|---:|---|
| F0 canonical TEST5 | 0.121212 | NOT MEASURED | 1.000 baseline | current 128-file projection 31.93 min |
| F2 M2 frozen PANNs head | 0.272727 | 0.406250 | Pearson 0.317627 / Spearman 0.277689 | +3.15 min / 1200 PROJECTED |
| F0 + 25% F2 | 0.090909 | NOT MEASURED | complementary blend | +3.15 min / 1200 PROJECTED |

All FILE values above are `MEASURED_GENERATOR_DISJOINT`. The 25% blend was inspected on the same split and is therefore exploratory, not independent calibration and not adoptable. Its apparent FILE contribution is `+0.015152 ADS`; F2 alone contributes `-0.075758 ADS` versus canonical F0.

Error overlap at each detector's empirical EER threshold:

- Both correct: `83`
- Both wrong: `3`
- Only canonical wrong: `13`
- Only F2 wrong: `33`

## PARTIAL / MIXED MODEL DIAGNOSTIC

Using generator-disjoint roots with deterministic virtual rendering:

- M1 partial FILE EER: `0.625000`; mixed RR-vs-fake FILE EER: `0.500000`
- M2/F2 partial FILE EER: `0.406250`; mixed RR-vs-fake FILE EER: `0.486111`
- Per-state EER is unavailable because RR/RF/FR/FF states are individually single-class; threshold-0.5 error and mean score are retained in the candidate JSON.
- These are synthetic stress diagnostics, not source-disjoint evidence.

## ADS CONTRIBUTIONS

- M1 Music: `-0.018750`
- M2 Music: `0.000000`
- F2 alone versus canonical FILE: `-0.075758`
- Same-split F0+F2 exploratory blend: `+0.015152`
- Voice: `0.000000` because Voice was not changed or evaluated in this iteration

## RUNTIME

| Files | Frozen TEST5 sec/file | Current x1 sec/file | Current median batch sec/file | Relative speedup | Current 1200-file projection |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.636262 | 1.443451 | 1.398542 | 1.1336x | 28.87 min |
| 128 | 1.590705 | 1.596626 | 1.633484 | 0.9963x | 31.93 min |

- Prediction max absolute difference: `5.46e-08` (`PASS <= 1e-6`)
- The 128-file result shows speed parity, not a reliable x1 speedup.
- Official TEST5 runtime remains `30m52s`; official x1 runtime is `NOT RUN`.
- Persistent executor remains rejected.
- DF-Arena batch-only throughput improved from `0.7697` sec/file at batch 16 to `0.7280` at batch 64 locally, but the submission batch stays 16 because flattening up to 192 PANNs/specialist crops at batch 64 has not passed the L4 memory gate.
- A CUDA-only overlap experiment is implemented behind explicit `gpu_overlap_enabled=true`, but the selected config leaves it disabled. CPU execution and default CUDA execution remain sequential. Unit/regression behavior is measured; official L4 wall-clock improvement is `NOT RUN`, so the experiment is not applied.
- Default-path end-to-end parity against frozen TEST5 on 8 direct files is `2.98e-08` (`PASS <= 1e-6`). Its tiny-sample wall clock is not used to claim a speed change.

## CPS

- TEST5: `0.9366803175` (`USER_REPORTED_PUBLIC`)
- Candidate production CPS: `NOT RUN`
- PANNs presence/fusion selected artifacts: `UNCHANGED`

## BOOTSTRAP

`NOT RUN`. No candidate has passed source-disjoint validation or adoption data gates.

## FINAL

`NOT ACQUIRED / NOT SEALED / NOT READ / NOT RUN`.

## NEXT BOTTLENECK

`MUSIC source diversity and deployable representation transfer` is now the single next bottleneck. M3 proves that a forensic-residual representation can reduce generator-disjoint Music EER (`0.3125 → 0.125`), but the selected lightweight model cannot yet capture that gain and M3 itself fails stability/runtime/license/source-disjoint gates. Public per-component EER is unavailable, so this is not presented as an official leaderboard decomposition.

## DECISION

`KEEP_TEST5`

M1 and M2 did not improve Music. M3 demonstrated a large non-final generator-disjoint gain but was correctly held out of production. F2 showed low error correlation and a same-split blend signal. Adoption remains blocked by the missing second approved paired Music source, source-disjoint validation, sealed globally unused final, independent fusion calibration, bootstrap, runtime, numerical-stability, and license gates.

Full repository regression: `172 passed in 42.52s`.
