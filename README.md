# Deep Voice Crime AI Detection - DACON (Updated 2026-08-28)

## Baseline Reference
Inspired by DACON codeshare 14153: PANNs + HTDemucs + DF_Arena_1B
- PANNs: Audio tagging for presence
- HTDemucs: Source separation (vocals/music)
- DF_Arena_1B: 1B deepfake detection (ONNX Int8)

Our solution integrates these ideas with lightweight alternatives for offline L4 inference.

## Architecture
- **Separation**: `librosa.effects.hpss` (lightweight HTDemucs alternative, harmonic=vocals, percussive=music) – `script.py:separate_vocals_music`
- **DF_Arena_1B**: `model/df_arena/df_arena_1b_int8.onnx` (1.37GB) via `onnxruntime` – FILE_FAKE primary
- **AASIST**: `src/models/aasist.py` base32 0.57M (2.3MB) – VOICE_FAKE, MUSIC_FAKE, presence
- **SpecCNN**: `src/models/beats_backbone.py` MelSpectrogram+CNN14 style – MUSIC fallback
- **Fusion**: `file = 0.4*DF_Arena +0.3*AASIST +0.3*probabilisticOR` – `script.py:infer_file`
- **Silence**: RMS<0.008 → present 0.02, fake 0.05 (fixed)
- **Sample**: respects `sample_submission.csv` order

## Quick Start
```bash
python script.py  # reads ./data/test (wav/mp3/flac mono/stereo) -> ./output/submission.csv
```

## Training (synthetic demo)
```bash
python scripts/run_all_stages.py  # 100 train, 30 val, 2epoch, VAL-A 0.72
# Real data: LibriSpeech/ASVspoof/WaveFake/FMA/FakeMusicCaps -> 100k+ samples
```

## Performance
- 5 files 2s (with DF_Arena 1.89s/file on CPU, 0.5s without), 1200 files ~38min CPU, ~10min on L4
- VRAM <4GB (DF_Arena ONNX CPU, AASIST 0.57M)
- Silence correctly 0.02, sample order respected, offline HF_HUB_OFFLINE=1
