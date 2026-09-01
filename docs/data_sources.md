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
| Project procedural v8 | `tools/procedural_audio_v8.py` | Project-authored numeric synthesis with no corpus, text, MIDI, pretrained model, sample, or web input | Project-owned output; organiser eligibility remains the final authority | 930 generated fake files in a separate candidate manifest |
| MLAAD-tiny v9 candidate | https://huggingface.co/datasets/mueller91/MLAAD-tiny (pinned revision `9143e5e`) | Fake audio CC BY-NC 4.0; paired originals retain the M-AILABS corpus notice | Attribution/non-commercial conditions apply; audio is not committed | 120 matched train real/fake pairs across 12 TTS/VC families; 40 disjoint unseen-validation pairs across 4 families |
| SONICS v9 candidate | https://github.com/awsaf49/sonics | Dataset CC BY-NC 4.0 | Attribution/non-commercial conditions apply; audio is not committed | 320 vocal synthetic songs across five Suno/Udio model generations |
| GuitarSet v9 candidate | https://zenodo.org/records/3371780 | CC BY 4.0 | Attribution required; audio is not committed | 360 real microphone recordings from 6 players |
| MTG-Jamendo v9 candidate | https://github.com/MTG/mtg-jamendo-dataset | Per-track Creative Commons licenses; metadata code/repository terms are separate | Only tracks with an explicit CC license permitting derivatives are accepted | 350 real low-bitrate tracks from 294 artists; checksum-pinned official shard |
| Echoes + FMA paired v9 candidate | https://huggingface.co/datasets/Octavian97/Echoes and https://github.com/mdeff/fma | Echoes generated audio CC BY-SA 4.0; each original FMA track retains its artist-selected CC license | Unknown and NoDerivatives FMA rows are rejected; attribution/ShareAlike/non-commercial terms vary per pair | 55 content-matched groups, 10 train and 2 unseen fake generators, 295 deduplicated rows |

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
| ArtifactNet v9.4 ONNX candidate | https://huggingface.co/intrect/artifactnet/tree/7c9b753a9d006b48e4bfaf85bf0157e135f4aad4 | CC BY-NC 4.0, research/non-commercial only; upstream patent notice says no patent license is granted | **Candidate evaluation only, not selected or bundled.** Generator-disjoint Music EER improved, but numerical stability, runtime, source-disjoint evidence, and competition-use approval gates failed |

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

The v7 audit found `APPROVED=500`, `REVIEW_REQUIRED=2,285`, and `REJECTED=0`.
The approved rows are all LibriSpeech real voice, so they do not contain the
fake/component diversity needed to form metric-complete training and
validation splits. Consequently `best_approved_only.pt` is **NOT RUN** rather
than being produced from an invalid one-class experiment. No unresolved row is
automatically promoted to approved status.

## Public diversity v9 candidate

`scripts/prepare_public_diversity_v9.py` pins MLAAD-tiny to one repository
revision and downloads only exact real/fake content pairs. The split group is
the original utterance identity, not the generator or rendered filename. The
12 training generators and four unseen-validation generators have zero content
group overlap. This directly removes the old voice shortcut in which real and
fake labels identified different source corpora.

The same script extracts five official SONICS generator families. Together with
the existing MusicGen family and ten training Echoes generators, the candidate
TRAIN pool contains 16 non-mix music generator families (AudioLDM2 remains an
existing held-out family).
SONICS real tracks are represented only by YouTube
IDs and are deliberately not downloaded; doing so would bypass the repository's
audio-distribution and per-track rights boundary.

`scripts/prepare_real_music_v9.py` adds two independent real-music sources. It
verifies the official GuitarSet archive size, the official MTG-Jamendo shard
and per-track hashes, and rejects missing, unknown, or NoDerivatives licenses.
FMA support remains optional because the full official archive is large; FMA
tracks are likewise filtered by their artist-selected per-track licenses.

`scripts/prepare_echoes_paired_v9.py` pins and hashes the Echoes archive, resolves
its exact `title - artist` reference against official FMA metadata, and downloads
only the matching FMA originals from official storage. Every accepted group has
both its real original and at least one generated counterpart. Ten generator
families are training candidates; DiffRhythm and SongGen plus their original
content are isolated in VAL-B. Unknown, ambiguous, and NoDerivatives originals
are rejected before acquisition.

All v9 artifacts are candidates only. They do not overwrite the selected v7
checkpoint, do not read the v6 final holdout, and are not included in
`submit.zip`. CC BY-NC data assumes the DACON run is non-commercial; this record
is not legal advice and organizer rules remain authoritative.

## Project-created procedural v8 candidate

`data/generated_v8/manifest.csv` describes 930 locally generated fake files
(840 candidate-training, 90 unseen-generator stress). The 160.4 MB of WAV data
is reproducible from seed `23674908` and intentionally ignored by Git. A second
full regeneration produced the identical manifest SHA256
`91ca2d0209449cca2627af0c4b2784b1f090c40f153613e14ddd8677c77493d8`.

The audit found no quality failures, exact duplicates, cross-split spectral
near-duplicates, or original/speaker/group/generator-family overlap. Generated
audio remains outside the selected v7 model until a controlled VAL-A/B/C/D
training comparison demonstrates a robust improvement. Full design and
reproduction details are in `docs/procedural_v8_dataset.md`.
