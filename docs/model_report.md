# Model report — 2026-08-31

## Executed work

- Official metric parity tests: PASS.
- DF-Arena label order and exact 64,600-sample input tests: PASS.
- Mix-label/balance/mode, base-source leakage, format loading, exact ID,
  offline/fail-fast and stale-fusion tests: PASS (16 tests total).
- Five CPU epochs for each SpecCNN specialist: RUN.
- HTDemucs training/evaluation: NOT RUN (not installed locally).
- New DACON submission score: NOT RUN.

Checkpoint selection is `0.55 * mean(VAL-A..D official score) + 0.25 * worst
split + 0.20 * VAL-B`.

| Specialist | Epoch | Composite | VAL-A | VAL-B | VAL-C | VAL-D |
|---|---:|---:|---:|---:|---:|---:|
| Voice SpecCNN | 5 | 0.81388 | 0.78728 | 0.73854 | 0.77641 | 0.64278 |
| Music SpecCNN | 1 | 1.00000 | 0.77918 | 0.84114 | 0.78011 | 0.77561 |

These are specialist diagnostics, not a claimed ensemble or DACON score. The
old DACON result (0.46623 total / 0.42838 ADS / 0.80685 CPS) belongs to the
previous archive.

## Controls

- DF-Arena logits `[spoof, bonafide]`, fake index 0, dynamic INT8 CPU graph.
- Pretrained PANNs AudioSet output is used; random DACON heads are not evidence.
- Checkpoint task/backbone metadata checked with strict state loading.
- No presence-to-fake gating; no hard 0.01/0.99 clipping.
- Calibration and submission call the same fusion implementation.
- Missing artifacts, requested-but-unavailable separation and empty input fail.

Exact-path 240-sample calibration selected `w_voice_file=0`,
`w_music_file=0.5`, `w_prob_or=0.5`, `w_df_arena=0.5`,
`w_df_component=0.25`, and `w_panns_presence=0.75`; composite calibration score
was 0.85115. Per-split official scores were VAL-A 0.85298, VAL-B 0.85924,
VAL-C 0.87196 and VAL-D 0.83606. These are subset validation results, not DACON
leaderboard results.

Steady-state local CPU benchmark: 20 repeated mixed-format fixture paths in
10.0267 s (0.5013 s/file), giving a linear 1,200-file projection of 10.0267 min.
This is not a full unique-file benchmark and does not include process/model-load
time. Official L4 benchmark: **NOT RUN**.

Final `submit.zip`: 928,326,895 bytes; SHA-256
`EDED111B394119FC2E8E7C89E69AAD08EFE88D7624DC75313920FA8A4286FA3A`.
Fresh extraction/default execution validator: PASS.
