# V13B Submission Readiness

Current decision: **KEEP_TEST5**. V13B is a dataset pilot, not a trained or submission-ready model.

- [x] DACON rules respected
- [x] no hidden test training, tuning, normalization fitting, ranking, or pseudo-labeling
- [x] selected TEST5 model/artifact hashes frozen
- [x] DATASET_V13 historical failed pilot preserved
- [x] acquired V13B files and provenance stored outside Git
- [x] direct production rows explicitly APPROVED
- [x] source-shortcut hard gate passed on content-grouped folds
- [x] TRAIN/VAL/CAL content, generator, near-duplicate, and SHA isolation passed
- [x] mixed/partial base and parent ancestry stored; development overlap audit passed
- [x] strict official metric schema and NaN/Inf fail-fast tests passed
- [x] CWD-independent artifact paths and sample/audio ID precheck passed
- [x] identity segment-plan input parity passed (three scans reduced to one)
- [x] partial-fake positives have label-matched real controls
- [x] RR/RF/FR/FF core is balanced
- [ ] second independent approved paired music source
- [ ] approved metric-complete source-disjoint validation
- [ ] completely unused FINAL_HOLDOUT_V13B acquired, history-audited, and sealed
- [ ] V13B model trained
- [ ] robust FILE/MUSIC/VOICE metrics measured
- [ ] bootstrap adoption gate passed
- [ ] new candidate runtime measured below 60 minutes
- [ ] V13B clean archive layout validated
- [ ] V13B offline inference smoke test passed

TEST5 remains recoverable byte-for-byte from `archive/pre_v13_selected/submit.zip`.
No V13B model, fusion file, `script.py`, selected checkpoint, or `submit.zip` was created or replaced.
