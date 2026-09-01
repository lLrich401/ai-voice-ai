# V13B Stage Report

## CURRENT SELECTED

branch: `x1`
development base commit: `e8434b9c368ee5de3368d8b0b04559cf19c3ffaa` (V13B work committed on descendant branch `x1`)
selected frozen submission: `TEST5`
TEST5 ADS: `0.6386349206` (OFFICIAL/USER-REPORTED)
TEST5 CPS: `0.9366803175` (OFFICIAL/USER-REPORTED)
TEST5 TOTAL: `0.6684394603` (OFFICIAL/USER-REPORTED)
runtime: `30m52s` (OFFICIAL/USER-REPORTED)

## DATASET STATUS

version: `DATASET_V13B_PILOT_20260901_1`
status: `DATASET NOT READY / MODEL NOT RUN`
train rows: `546` (386 paired core + 64 partial/control + 96 mixed)
cal rows: `112`
generator val: `132`
source-disjoint val: `NOT CREATED`
final holdout: `NOT CREATED / NOT SEALED / NOT RUN`
paired voice: `2` independent sources (`MLAAD`, `DFADD/VCTK`)
paired music: `1` independent source (`Echoes/FMA`)
partial: `64` rows across 2/5/10/20/30/50/70/100% occupancy, half real controls
mixed: `96` rows; RR/RF/FR/FF = `24/24/24/24`

## SOURCE SHORTCUT

metadata AUC: `0.5816531987`
acoustic AUC: `0.6413594996`
combined AUC: `0.7181937770`
decision: direct paired-core shortcut hard gate `PASS`; whole V13B remains blocked by structural data gates
source prediction from shallow acoustics: `0.7610101010` balanced accuracy (chance `0.3333`)
largest remaining label fingerprints: duration, ZCR, RMS, spectral centroid, silence, HF ratio

## DATA QUALITY

exact duplicate SHA rows: `0`
near-duplicate/content pairs: `315`; every pair contains real and fake in one role
TRAIN/VAL/CAL split identifier leakage: `0` for content, split group, near-duplicate, processed SHA, and source SHA
license unresolved: second paired music source, future completely unused final source, and all REVIEW_REQUIRED candidates
canonical policy: label-independent 16 kHz mono PCM16 with the same peak ceiling; original and processed SHA retained

## CURRENT BOTTLENECK

1. A second independent, row-license-resolved paired music domain is missing.
2. Without it, an approved metric-complete source-disjoint validation set cannot be isolated.
3. A globally unused source for sealed FINAL_HOLDOUT_V13B has not been acquired.

## BEST MUSIC

architecture: `NOT RUN`
music EER: `NOT MEASURED`
unseen music EER: `NOT MEASURED`

## BEST FILE

architecture: `NOT RUN`
file EER: `NOT MEASURED`
partial file EER: `NOT MEASURED`
unseen file EER: `NOT MEASURED`

## BEST VOICE

architecture: `NOT RUN`
voice EER: `NOT MEASURED`
worst voice EER: `NOT MEASURED`

## FINAL CANDIDATE

FILE EER: `NOT RUN`
VOICE EER: `NOT RUN`
MUSIC EER: `NOT RUN`
ADS: `NOT RUN`
CPS: `NOT RUN`
TOTAL: `NOT RUN`

## BOOTSTRAP

win rate: `NOT RUN`
ADS p05: `NOT RUN`
ADS median: `NOT RUN`
ADS p95: `NOT RUN`

## RUNTIME

projected: `NOT PROJECTED for V13B`
submission-like: `NOT RUN for V13B`; selected TEST5 official runtime is `30m52s`

## FINAL HOLDOUT

status: `NOT CREATED / NOT SEALED`
result: `NOT RUN`

## SUBMISSION

decision: `KEEP_TEST5`
zip: frozen `archive/pre_v13_selected/submit.zip`
sha256: `5898ad19b5c92e46d54aca529336b99f9e17838806766c654a3585051388fdcb`
validator: frozen TEST5 `PASS`; V13B `NOT RUN`
offline smoke: frozen TEST5 `PASS`; V13B `NOT RUN`

## NEXT BEST ACTION

Acquire a second paired music corpus with at least 10 content groups and explicit row-level competition permission. Do not relabel ordinary codec/DSP transformations as AI fake. After that, reserve a completely unused source for FINAL_HOLDOUT_V13B, rerun the same frozen shortcut policy, and only then start the three-candidate Music architecture pilot.
