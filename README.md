# Deep Voice Crime AI Detection - DACON 236749 (Updated 2026-08-28) — Baseline 14153 Aligned

## Baseline Reference
DACON codeshare 14153: **PANNs·HTDemucs·DF_Arena_1B 기반 AI 탐지** (`https://dacon.io/competitions/official/236749/codeshare/14153`)
- **PANNs CNN14** (qiuqiangkong/audioset_tagging_cnn, Cnn14_mAP=0.431, 81M, 527 AudioSet classes) – Speech/Music tagging → VOICE_PRESENT / MUSIC_PRESENT
- **HTDemucs** (facebookresearch/demucs, Hybrid Transformer) – 4-stem separation (drums/bass/other/vocals) → vocals vs music, handles 혼합 (speech+music) case
- **DF_Arena_1B** (pranjal-pravesh/df_arena_1b, 1B params, ONNX Int8 1.37GB) – deepfake detection → FILE_FAKE primary

Our solution reproduces baseline with offline L4 optimizations.

## Architecture (script.py)
- **Separation** `src/models/demucs_wrapper.py:HTDemucsSeparator` – tries `demucs.api.Separator(model=htdemucs, device=cuda/cpu)`, fallback to `librosa.effects.hpss` (harmonic=vocals, percussive=music) for offline/fast. RMS check adjusts presence if separated energy <0.15×orig.
- **PANNs** `src/models/panns.py:PANNsPresenceWrapper` – CNN14 via torchaudio Mel (no torchlibrosa dep). Loads `model/panns/Cnn14_mAP=0.431.pth` if present; blends 0.7×AudioSet Speech/Music max +0.3×learned head. If no pretrained, falls back to AASIST presence (avoids random).
- **DF_Arena_1B** `model/df_arena/df_arena_1b_int8.onnx` via onnxruntime – session options 4 threads, ORT_ENABLE_ALL, CUDA→CPU fallback. Input `[1,64000]` (4s 16k), softmax fake=prob[1], 5-seg topk_mean aggregation.
- **AASIST** `src/models/aasist.py` base32 0.57M – SincConv 20→ResNetSE×4→BiGRU→Attention, 5 heads (file/voice/music_fake, voice/music_present). Loaded from `model/best.pt` (tries base32/16/64).
- **SpecCNN** `src/models/beats_backbone.py` – Mel128 + CNN14-style fallback for music (used in training FusionModel).
- **Fusion** `script.py:infer_file`:
  - Silence: RMS<0.008 → `[0.05,0.05,0.05,0.02,0.02]`
  - Presence: `voice_present = 0.6*PANNs +0.4*AASIST` (if PANNs pretrained) else AASIST; then RMS-adjusted
  - Fake calibration: if `present<0.4`, `fake = present*fake_raw + (1-present)*0.05` (suppress hallucination when source absent)
  - File: `file = 0.4*(0.5*DF+0.5*AASIST_file) +0.3*probOR(voice_fake,music_fake)+0.3*AASIST_file` (with low-presence conservative branch); no DF → `0.6*AASIST+0.4*probOR`
  - Clip to `[0.01,0.99]`
- **Segmentation** `src/preprocess.py:extract_segments` – 4s uniform5 (0,0.25,0.5,0.75,1.0), topk_mean(k=2) for fakes, mean for presence.
- **Sample order** respects `sample_submission.csv` first column (recursive search: ./, ./data/, etc.) and handles stem vs filename.

## Quick Start
```bash
pip install -r requirements.txt  # torch+torchaudio+librosa+soundfile+onnxruntime
python script.py  # reads ./data/test (wav/mp3/flac/m4a/ogg mono/stereo) -> ./output/submission.csv
python tools/validate_submission.py submit.zip  # checks offline zip
```

## Training
```bash
python scripts/run_all_stages.py  # synthetic 100/30, 2ep, VAL-A ~0.72 (real data would be higher)
# Real data example:
python scripts/download_datasets.py --datasets librispeech_dev wavefake fma_small --max_samples 1000
# Then train with src/train.py (AMP, cosine, early stopping) on LibriSpeech/ASVspoof/WaveFake/FMA/FakeMusicCaps
```

## Performance
- 5 files ~2-4s (with DF_Arena 1.89s/file CPU, 0.5s without PANNs), 1200 files ~38min CPU, ~10min L4, <60min limit
- VRAM <4GB (DF_Arena ONNX CPU, AASIST 0.57M, PANNs 81M optional but CPU), L4 22.4GB OK
- Silence correctly 0.02, sample order respected, offline HF_HUB_OFFLINE=1, handles 4s-1min, telephone, mp3/wav/flac.

## Baseline Alignment Notes
- Original baseline code not downloadable without login; this reproduction follows title & competition spec (PANNs for presence, HTDemucs for separation, DF_Arena for fake) with lightweight fallbacks to satisfy code-submission offline & time limits (demucs heavy → hpss fallback, PANNs pretrained optional, DF_Arena ONNX already included).
- Improvements over naive hpss-only: proper HTDemucs wrapper with demucs.api support, real PANNs CNN14 implementation (not just SpecCNN), presence-aware fake suppression, RMS-based presence adjustment, robust sample handling.
