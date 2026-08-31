# Data and model provenance

This is a provenance record, not legal advice and not a relicensing of any
third-party artifact. Training audio is not included in `submit.zip`.

| Data used | Source | License / use limits | Redistribution | Local composition |
|---|---|---|---|---:|
| LibriSpeech | https://www.openslr.org/12 | CC BY 4.0; commercial use is allowed with attribution | Allowed under CC BY 4.0 with notices | 500 real voice |
| ASVspoof 2019 LA mirror | https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA and https://www.asvspoof.org/ | Official corpus has a usage agreement oriented to research/development; the mirror does not establish broader rights | **Do not redistribute from this project; verify the signed corpus terms** | 121 bona fide, 379 spoof (A01/A02/A03) |
| WaveFake audio mirror | https://huggingface.co/datasets/ajaykarthick/wavefake-audio; upstream https://doi.org/10.5281/zenodo.5642694 | Upstream WaveFake is CC BY-SA 4.0; its datasheet also notes source-specific redistribution constraints/research expectations | ShareAlike/attribution apply; verify that the mirror preserves upstream notices | 62 real/R, 438 fake (WF1–WF7) |
| GTZAN mirror | https://huggingface.co/datasets/sanchit-gandhi/gtzan | The inspected mirror does not provide a sufficiently clear audio redistribution/commercial license | **Unknown; do not redistribute** | 985 byte-unique real-music tracks after repair, 10 genres |
| MusicGen outputs | https://huggingface.co/facebook/musicgen-large | Model weights CC BY-NC 4.0; non-commercial restriction. Output rights may also depend on prompts/source material | Do not assume commercial or unrestricted redistribution | 150 fake music |
| AudioLDM2 outputs | https://audioldm.github.io/audioldm2/ | Exact checkpoint/output license used by the historical generator run was not recorded | **Unknown; do not redistribute until verified** | 150 fake music |

All local audio was decoded to mono 16 kHz. Training adds common random gain,
noise, EQ/band-limit, reverb, clipping and dynamic-range transforms regardless
of source. Split-internal mixed samples add multiple SNRs, simultaneous,
sequential, partial-overlap and crossfade layouts. VAL-C and VAL-D add codec and
telephone simulations respectively.

## Bundled pretrained models

| Model | Source/version | Recorded terms | Use here |
|---|---|---|---|
| DF-Arena 1B INT8 | https://huggingface.co/pranjal-pravesh/df_arena_1b | Model card declares MIT | Offline inference, original waveform, 64,600 samples |
| PANNs CNN14 16 kHz (`Cnn14_16k_mAP=0.438.pth`) | https://github.com/qiuqiangkong/audioset_tagging_cnn and https://zenodo.org/records/3987831 | Preserve repository/checkpoint notices | Official 16 kHz AudioSet frontend: 512 FFT, 160 hop, 64 mel, 50–8000 Hz |
| HTDemucs | https://github.com/facebookresearch/demucs | MIT repository; weights may have separate provenance | Optional; not installed or evaluated locally |

The root MIT license covers only repository-authored code/documentation. It
does not cover datasets, pretrained weights, derived checkpoints, or generated
media. In particular, unresolved GTZAN/AudioLDM2 and restricted ASVspoof terms
must be reviewed before any commercial use or redistribution.

`data/manifest.csv` records source URL, version, license status,
competition-use review state, redistribution/commercial restrictions, original
ID, speaker/generator IDs, exact content SHA256, near-duplicate group, and split
group for every original file. `python scripts/enrich_manifest_provenance.py`
rebuilds and validates those fields. Training accepts
`--require_approved_provenance` to exclude every `REVIEW_REQUIRED` row. Current
checkpoints predate that strict filter and retain the review caveats above.
