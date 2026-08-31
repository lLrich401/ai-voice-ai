# v9 public-diversity experiment

Selection boundary: VAL-A/B/C/D only. The existing v6 final holdout is sealed
and must not be read, copied, or evaluated.

Implemented candidate inputs:

- MLAAD-tiny exact real/fake pairs: 12 training TTS/VC families and four
  generator-disjoint VAL-B families.
- SONICS: five Suno/Udio generator versions and 320 vocal synthetic songs.
- GuitarSet and MTG-Jamendo real-music acquisition with license/hash gates.
- Echoes/FMA: 55 exact real/generated content groups, ten training generators
  and two content-disjoint unseen music generators for VAL-B.
- AASIST voice training on a detected GPU only (CUDA or Intel XPU).
- `voice_channel_v9`: mutually exclusive channel families plus moderate
  loudspeaker-room-microphone simulation, resampling and short dropouts.
- TRAIN-only partial fake generation; validation sources are rejected.
- Decode-quality audit over all 1,645 public-candidate files (duration, silence,
  clipping, sample rate and channels); see `public_audio_quality.json`.
- Label-first sampling: real/fake mass is balanced before source/generator
  balancing, preventing many fake generator IDs from overwhelming one original
  source.

Fusion recalibration is conditional: it is run only if the new voice candidate
beats the selected v7 model on the robust cross-domain objective without a
VAL-B, VAL-C, or VAL-D regression. Until then the selected v7 pipeline and its
fusion remain unchanged.

Reproduction order:

```powershell
python scripts/prepare_public_diversity_v9.py
python scripts/prepare_real_music_v9.py
python scripts/prepare_echoes_paired_v9.py --per_generator 20
python scripts/prepare_v9_candidate_splits.py
python scripts/audit_public_audio_v9.py
python scripts/train_aasist_gpu_v9.py --epochs 8 --batch_size 24 --partial_fake_count 1000
python scripts/train_music_gpu_v9.py --epochs 8 --batch_size 32
```
