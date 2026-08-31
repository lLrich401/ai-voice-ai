# v10 VAL-C / VAL-D audit

Status: measured non-final validation. Final holdout: **NOT RUN**.

## Diagnosis

- Historical VAL-C applies a deterministic 3.5 kHz low-pass despite the `codec_mp3` name. It does not cover codec, quantization, or resampling diversity.
- Historical VAL-D is the worst selected domain, but its failure is content-dependent. On the old difficult cache, telephone processing lowered the mean DF fake score by 0.192 while the voice specialist moved by only 0.022. On a disjoint 192-content confirmation set, ordinary μ-law, A-law, and narrow-band telephone transforms did not reproduce an EER regression.
- A proposed channel-dependent DF weight improved the old VAL-D cache but was rejected because it did not generalize as a consistent telephone rule.
- The first low-bit proxy exposed a validation bug: unsigned quantization did not preserve zero and added DC to silence. Signed zero-preserving quantization replaced it, tests were added, and quantized profiles were rerun.

## Independent channel stress (192 rows/profile)

| Profile | FILE EER | VOICE EER | MUSIC EER | CPS | TOTAL |
|---|---:|---:|---:|---:|---:|
| clean | 0.07331 | 0.07468 | 0.00000 | 0.99121 | 0.95269 |
| codec 3.5 kHz LP | 0.05785 | 0.05276 | 0.00000 | 0.99129 | 0.96360 |
| codec 12 kHz + 12-bit | 0.05785 | 0.06818 | 0.00000 | 0.99124 | 0.96082 |
| codec 8 kHz + 8-bit | 0.08878 | 0.07468 | 0.00833 | 0.99071 | 0.94343 |
| telephone μ-law | 0.05785 | 0.05276 | 0.00000 | 0.98885 | 0.96336 |
| telephone A-law | 0.05785 | 0.05276 | 0.00000 | 0.99114 | 0.96358 |
| telephone narrow | 0.05785 | 0.05276 | 0.00000 | 0.98752 | 0.96322 |
| telephone low-bit proxy | 0.11970 | 0.14286 | 0.00000 | 0.98417 | 0.91884 |

The low-bit telephone condition is a deterministic robustness proxy, not bit-exact GSM.

## Model experiments

| Candidate | VAL-A VOICE EER | VAL-B | VAL-C | VAL-D | Decision |
|---|---:|---:|---:|---:|---|
| selected v7 | 0.14583 | 0.14583 | 0.16667 | 0.20833 | keep |
| v10 scratch | 0.14583 | 0.18750 | 0.16667 | 0.22917 | reject |
| v10 fine-tune | 0.14583 | 0.16667 | 0.16667 | 0.18750 | reject: VAL-B regression |
| interpolation 0.60 | 0.14583 | 0.16667 | 0.16667 | 0.18750 | reject: same trade-off |

No candidate satisfied the predeclared no-domain-regression rule. The selected checkpoint and fusion therefore remain unchanged. `latest_results.json` was not rewritten because there is no selected pipeline change and the protected final holdout was not run.

## Implemented improvements

- deterministic multi-profile VAL-C/VAL-D stress evaluator with source-path fail-fast;
- codec profiles covering bandwidth, resampling, and signed quantization;
- telephone profiles covering μ-law, A-law, narrow band, and a clearly labeled low-bit proxy;
- controlled `voice_channel_v10` training augmentation using one primary channel per sample;
- strict checkpoint initialization and epoch-zero preservation for safe fine-tuning;
- strict single-model weight-interpolation experiment tooling;
- deterministic, silence-preserving, shape/range tests.

## Verification

- pytest: 77 passed in 18.97 s
- bundled runtime source: byte-identical to working `src`
- `submit.zip`: 935,630,022 bytes (892.29 MiB)
- SHA256: `885156505bd7ad8a1a0632ffe83799b5c982093b08e4256c7e7cd9ccd4da44e0`
- archive layout and mandatory artifacts: PASS
- offline smoke inference: 3 files in 5.4 s, PASS
- official DACON server runtime: NOT RUN
