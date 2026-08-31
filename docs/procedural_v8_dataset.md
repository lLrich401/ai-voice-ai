# Procedural v8 fake-audio dataset

## Purpose

This dataset is a separate v8 robustness candidate. It does **not** replace the
selected v7 pipeline and it is not included in `submit.zip`.

Every waveform is created locally from numeric seeds by
`tools/procedural_audio_v8.py`. The generator reads no speech corpus, music,
lyrics, MIDI, impulse response, pretrained checkpoint, web resource, or other
media asset. It does not call a network service. Generated audio is labelled
only as synthetic/fake; no procedural sample is represented as real audio.

This design minimizes third-party copyright and licence dependency; it does not
guarantee a legal conclusion. It is not legal advice, and the competition
organiser remains the authority on dataset eligibility. The complete generation seed, generator version, file hash, source
code location, and provenance decision are stored per row.

## Content and split design

- Synthetic voice: abstract pseudo-phoneme/formant sequences. No text or real
  speaker recording is used.
- Synthetic music: seed-derived original pitch, rhythm, oscillator, and noise
  sequences. No known melody or audio sample is used.
- Mixed files: locally generated fake voice plus locally generated fake music.
- Train content has three voice and three music renderer families.
- Validation uses different content seeds, speaker IDs, grouping IDs, and
  renderer families. It is an unseen-fake stress set, not an EER validation set,
  because it intentionally contains no procedurally labelled real audio.
- All renders of one abstract content item share an original-content and
  near-duplicate group and remain in the same split.

Official five-head labels are:

| Content | FILE_FAKE | VOICE_FAKE | MUSIC_FAKE | VOICE_PRESENT | MUSIC_PRESENT |
|---|---:|---:|---:|---:|---:|
| voice only | 1 | 1 | 0 | 1 | 0 |
| music only | 1 | 0 | 1 | 0 | 1 |
| mixed | 1 | 1 | 1 | 1 | 1 |

## Reproduction

```powershell
python tools/generate_procedural_v8.py
python tools/audit_procedural_v8.py
python tools/prepare_v8_training_manifest.py
```

The defaults create 930 WAV files: 840 training rows and 90 unseen-generator
stress rows (about 160.4 MB with the current deterministic seed). Audio lives below `data/generated_v8/audio/` and is deliberately
ignored by Git. `data/generated_v8/manifest.csv`, the audit report, source code,
and documentation are reproducible and may be versioned.

The merge command writes only to `data/splits_v8_candidate/`. It never mutates
`data/splits/`, never uses calibration/final-holdout data, and never starts
training automatically.

## Measured risk decision

The initial v8.1 corpus is retained for diagnostics but is **not authorized for
training**. An audio-only logistic classifier separated it from existing fake
audio at AUC 1.0 for both voice and music, proving a severe procedural-source
fingerprint. It would also form 29.3% of candidate rows and about 24–25% of the
specialist sampler probability.

On the generator-disjoint 90-file stress set, selected v7 achieved 100% voice
fake recall but 0% voice-presence recall. It achieved 0% music-fake recall.
Those results show that the current waveforms do not yet match the semantic and
acoustic support of real-world generated voice/music. They are positive-only
stress measurements, not EER.

`tools/prepare_v8_training_manifest.py` now reads the measured risk report and
marks the candidate `REJECT_CURRENT_DATASET_HIGH_SOURCE_FINGERPRINT` with
`training_authorized=false`. A future v8.2 generator must reduce the audio-only
source-classifier AUC and then pass a controlled SpecCNN experiment on
VAL-A/B/C/D before adoption.
