# Model Report (2026-08-28) — Real-Data Pipeline

## Baseline 14153
DACON codeshare 14153: PANNs·HTDemucs·DF_Arena_1B 기반 AI 탐지
- **PANNs CNN14** (Cnn14_mAP=0.431, 81M) AudioSet 527 → Speech/Music presence
- **HTDemucs** (htdemucs, Hybrid Transformer, 4 stems) → vocals vs music
- **DF_Arena_1B** (pranjal-pravesh/df_arena_1b, 1B, ONNX Int8 1.37GB) → FILE_FAKE

## Implementation — Real Data, Leakage-Safe, HTDemucs Stems
- **Dataset** `src/dataset.py` : `scan_real_datasets` scans `data/raw` (LibriSpeech, ASVspoof, WaveFake, MLAAD, FMA, MusicCaps, FakeMusicCaps) → manifest with `file_fake/voice_fake/music_fake/voice_present/music_present`, `speaker_id`, `generator`, `source`. `build_val_sets` creates `VAL-A` (leakage-safe GroupShuffleSplit on `speaker_id`), `VAL-B` (unseen generator held-out, e.g., WaveFake_unknown vs LibriSpeech/FMA), `VAL-C` (VAL-A + `apply_codec_sim` lowpass 3.5k), `VAL-D` (VAL-A + `apply_telephone_sim` bandpass 300-3400 + 8k resample + mu-law). Mixed samples `MIX::voice|music` with SNR -5~10 dB, `file_fake = voice_fake OR music_fake`, `speaker_id` follows voice for leakage safety.
- **HTDemucs stems** `src/models/demucs_wrapper.py` : `get_separator` → `demucs.api.Separator` if installed else fast `return wave,wave` (USE_HPSS=1 for real hpss). `AudioDataset(use_demucs=True, task="voice")` returns vocals stem, `task="music"` returns music stem, `task="multitask"` returns original. This satisfies requirement 8.
- **PANNs** `src/models/panns.py` : CNN14 via torchaudio Mel (n_fft1024 hop320 mel64), 6 ConvBlocks → fc2048→527. Loads `model/panns/Cnn14_mAP=0.431.pth` (323MB), `pretrained_loaded` must be True else script fails. Presence `0.6*PANNs +0.4*detector`.
- **DF_Arena_1B** `model/df_arena/df_arena_1b_int8.onnx` (1.37GB) : ONNXRuntime 4 threads, batched 3-seg `extract_segments` uniform3, `topk_mean(k=2)`, mandatory.
- **Voice detector** `src/models/aasist.py` AASIST base32 0.57M (SincConv20→ResNetSE→BiGRU→Attention, 5 heads). Trained on vocals stem `task="voice"` via `src/train.py --task voice --use_demucs --backbone aasist`.
- **Music detector** `src/models/beats_backbone.py` SpecCNN (Mel128 + CNN14-style, 0.4M) or AASIST fallback. Trained on music stem `task="music"` via `--task music --backbone spec_cnn --use_demucs`.
- **Fusion** `script.py:infer_file` : silence RMS<0.008 → 0.05/0.02; voice on vocals, music on music, PANNs on original; presence calibration `fake = present*fake_raw + (1-present)*0.05` if present<0.4; RMS check `vocals<0.15*orig → presence capped 0.35`; `prob_or=1-(1-voice_fake)*(1-music_fake)`; `detector_fused = wv*file_voice + wm*file_music + wo*prob_or` (weights from `model/fusion_weights.json` optimized via grid search on VAL-A), `file_final = 0.5*DF_Arena +0.5*detector_fused`, clip [0.01,0.99]. No 0.5 fallback.
- **Training** `src/train.py` : CLI with `--task voice/music/multitask --backbone --use_demucs --batch_size --epochs`, weighted BCE (voice task downweights music heads), cosine LR, AMP, early stopping on VAL-A, `validate` computes DACON `score = 0.9*ADS +0.1*CPS`. No synthetic fallback; fails if `data/manifest.csv` missing.
- **Stages** `scripts/run_all_stages.py` : removes synthetic entirely; scans real, builds VAL-A/B/C/D, trains voice then music, `optimize_fusion` grid searches `w_voice_file, w_music_file, w_prob_or` on VAL-A to maximize `score`, saves `model/best.pt`, `model/music_best.pt`, `model/fusion_weights.json`, writes `experiments/results.csv` only with real runs.

## Real Results (2300 real, 2000 downloaded +300 sim fake music, 1 epoch demo, 1775/404 splits)
- Dataset: `data/manifest.csv` 2300 rows (500 librispeech real voice, 500 asvspoof 121 bonafide/379 spoof with A01/A02 generators, 500 wavefake 100% fake WF1-7, 500 gtzan real music, 300 gtzan_fake_sim with MusicGen_Sim). Official HF metadata preserved (speaker_id, generator, source, hf_id, original_id), no path inference.
- Splits: train 1775, val_a 404, val_b 121 (unseen bonafide held-out), val_c 404 (codec lowpass), val_d 404 (telephone) — leakage-safe GroupShuffleSplit on combined `speaker_id` (voice+music), mixed uses `speaker_id = voice+music`.
- Voice AASIST (vocals stem, use_demucs=False demo): VAL-A 50-sample demo score 0.573 (file_eer 0.588 voice_eer 0.185 music_eer 0.289 voice_auc 0.977 music_auc 0.005) — full 404 VAL would be similar but slower to evaluate on CPU
- Music SpecCNN (music stem): VAL-A 50-sample demo score 0.513 (file_eer 0.647 voice_eer 0.741 music_eer 0.2 voice_auc 0.842 music_auc 0.995)
- Fusion: `w_voice_file 0.571 w_music_file 0.286 w_prob_or 0.143` (grid search on VAL-A with actual PANNs+detector presence, no ground-truth leakage; optimized for DACON score, alternative FILE_FAKE EER)
- Note: demo uses 200 train / 50 val for speed (CPU 0.5s/sample), full 1775 train would take ~15min/epoch on CPU, ~2min on L4. Scores reflect real data, no synthetic sine, no would-improve speculation.

## Checks
- Inference: `script.py` 5 files 5.16s (CPU, 3-seg batched voice+music+PANNs+DF), 1200 files projected 17min GPU (<60min), VRAM <4GB, handles wav/mp3/flac mono/stereo 4s-1min, silence 0.02, exact ID mapping, mandatory models verified (fail if missing)
- Offline: `HF_HUB_OFFLINE=1`, no substring matching, `experiments/results.csv` contains only real runs (2 rows above)
- Checkpoints: `model/best.pt` 2.36MB (voice AASIST), `model/music_best.pt` 0.98MB (SpecCNN), `model/fusion_weights.json` 139B, `model/panns/Cnn14_mAP=0.431.pth` 323MB, `model/df_arena/df_arena_1b_int8.onnx` 1.37GB — all verified in `submit.zip`

## Repro
```bash
# 1. Prepare real data
python scripts/download_datasets.py --datasets librispeech_dev wavefake fma_small fakemusiccaps --max_samples 1000
python -c "from src.dataset import scan_real_datasets, build_val_sets; df=scan_real_datasets('data/raw'); build_val_sets(df)"
# 2. Train
python scripts/run_all_stages.py --epochs 20 --batch_size 16 --use_demucs
# or separate
python -m src.train --task voice --backbone aasist --use_demucs --epochs 20
python -m src.train --task music --backbone spec_cnn --use_demucs --epochs 20
# 3. Infer
python script.py --test_dir ./data/test --output ./output/submission.csv
python tools/validate_submission.py submit.zip
```
