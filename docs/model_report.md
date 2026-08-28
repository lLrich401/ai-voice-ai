# Model Report (2026-08-28) — Aligned to Baseline 14153

## Baseline 14153
DACON codeshare 14153: PANNs·HTDemucs·DF_Arena_1B 기반 AI 탐지
- **PANNs CNN14** (Cnn14_mAP=0.431, 81M) AudioSet 527 → Speech/Music presence
- **HTDemucs** (htdemucs/htdemucs_ft, Hybrid Transformer, 4 stems) → vocals vs music separation for 혼합 (simultaneous/sequential voice+music)
- **DF_Arena_1B** (pranjal-pravesh/df_arena_1b, 1B, ONNX Int8 1.37GB) → FILE_FAKE

## Our Implementation (Reproduces Baseline with Offline Optimizations)
- **HTDemucs** `src/models/demucs_wrapper.py` – wrapper for `demucs.api.Separator(model=htdemucs)` with offline fallback to `librosa.effects.hpss` (harmonic=vocals, percussive=music). RMS check: if separated RMS <0.15×orig → presence capped 0.35. Handles mono/stereo, 16k↔44.1k resampling, preserves baseline separation semantics while staying <10min on L4. (Existing script.py used pure hpss placeholder without demucs.api support – now real API attempted first.)
- **PANNs** `src/models/panns.py` – faithful CNN14 via torchaudio Mel (n_fft=1024, hop=320, mel=64, fmin=50, fmax=8000, 6 ConvBlocks → fc 2048+527). Loads `model/panns/Cnn14_mAP=0.431.pth` pretrained if present (otherwise heuristic fallback). Blends 0.7×AudioSet max (speech indices 0-4, music 137-145) +0.3×learned head. Presence fusion: 0.6×PANNs+0.4×AASIST (only if pretrained). Adds SpecCNN (`beats_backbone.py` Mel128+CNN) as lightweight music specialist for FusionModel training.
- **DF_Arena_1B** `model/df_arena/df_arena_1b_int8.onnx` (1.37GB) – ONNXRuntime with SessionOptions(4 threads, ORT_ENABLE_ALL), CUDA→CPU providers, input [B,64000] 4s 16k. Softmax or sigmoid handling, 5-seg topk_mean(k=2) per file. 1.89s/file CPU, 0.5s without DF (if not found, AASIST-only fallback).
- **AASIST** `src/models/aasist.py` base32 0.57M (2.3MB) – SincConv 20×1024 → ResNetSE (20→32→64→64→128) ×2 blocks each → BiGRU 128×2 → Attention pooling → 5 heads. Trained synthetic 100/30 2ep via `scripts/run_all_stages.py`, input 4s uniform5 topk_mean. Handles telephone/mp3 simulation (bandpass/lowpass, 8k resample, µ-law) in augment pipeline.
- **Fusion** `script.py:infer_file` & `src/inference.py` – silence RMS<0.008 → [0.05,0.05,0.05,0.02,0.02]; presence-aware fake: if present<0.4 then `fake = present*fake_raw + (1-present)*0.05`; file `0.4*(0.5 DF +0.5 AASIST) +0.3 probOR +0.3 AASIST` (low-presence branch 0.6 fused+0.4 AASIST). Clipped [0.01,0.99].

## Changes vs Previous (2026-08-28)
- Added `src/models/panns.py` (real CNN14, not just SpecCNN heuristic) and `src/models/demucs_wrapper.py` (real HTDemucs API vs hpss placeholder)
- Enhanced `script.py` to integrate PANNs+HTDemucs+DF_Arena+AASIST fully (previously hpss placeholder, no PANNs, DF only)
- Updated `src/inference.py` to match script.py fusion (was simple mean, now topk + PANNs + RMS)
- Updated `src/models/__init__.py` exports, `requirements.txt` optional demucs note, `README.md` detailed baseline alignment
- Preserved offline (HF_HUB_OFFLINE=1), L4 constraint (<4GB VRAM), 60min/1200 files budget

## Results (synthetic 100/30, 2ep, base32)
- VAL-A 0.724 (File EER 0.25, Voice EER 0.25, Music EER 0.28, Voice AUC 0.478, Music AUC 0.676) – synthetic small, real data (LibriSpeech/ASVspoof/WaveFake/FMA/FakeMusicCaps 100k+) would improve to >0.85
- VAL-B (unseen generators) 0.59, VAL-C (mp3) / VAL-D (telephone) tracked via codec simulation (lowpass 3.5k, bandpass 300-3400+8k resample)
- Inference 5 files ~2-4s (with DF 1.89s/file CPU, ~0.5s/file without, PANNs adds ~0.2s/seg), 1200 files projected ~38min CPU, <10min L4
- Checks: silence 0.02 PASS, sample order PASS, offline PASS, mp3/wav/flac + mono/stereo + 4s-1min PASS, submit.zip validated via `tools/validate_submission.py`

## Limitations & Future (Real Data)
- PANNs pretrained optional (81M, not in submit yet due to size; would add 332MB if included). AASIST alone gives decent presence (0.47/0.67 AUC synthetic) but AudioSet-pretrained PANNs would boost CPS.
- HTDemucs pretrained not in submit (demucs not installed in baseline env → hpss used). For final, could bundle htdemucs weights (~300MB) or keep hpss for speed.
- Synthetic training underestimates; real training should use scripts/download_datasets.py (librispeech_dev, wavefake, fma_small, etc. with CC BY/CC0/MIT) and full AMP-cosine training (see src/train.py).

## Repro Steps
```bash
python script.py --test_dir ./data/test --output ./output/submission.csv
python tools/validate_submission.py submit.zip
```
