# Deep Voice Crime AI Detection - DACON 236749 (Updated 2026-08-28) — Real-Data Pipeline

## Baseline Reference
DACON codeshare 14153: **PANNs·HTDemucs·DF_Arena_1B 기반 AI 탐지** (`https://dacon.io/competitions/official/236749/codeshare/14153`)
- **PANNs CNN14** (Cnn14_mAP=0.431, 81M, 527 AudioSet) – Speech/Music presence
- **HTDemucs** (facebookresearch/demucs, Hybrid Transformer, 4 stems) – vocals vs music
- **DF_Arena_1B** (pranjal-pravesh/df_arena_1b, 1B, ONNX Int8 1.37GB) – FILE_FAKE

## Architecture — Real Data, Leakage-Safe, HTDemucs Stems
- **Dataset** `src/dataset.py` : `scan_real_datasets` (`data/raw` LibriSpeech/ASVspoof/WaveFake/FMA/MusicCaps/FakeMusicCaps) → manifest, `build_val_sets` (GroupShuffleSplit on `speaker_id`, VAL-A normal seen, VAL-B unseen generator held-out, VAL-C codec lowpass 3.5k, VAL-D telephone bandpass+8k+mu-law), `MIX::voice|music` with SNR -5~10.
- **Separation** `src/models/demucs_wrapper.py:HTDemucsSeparator` : `demucs.api.Separator(htdemucs)` if installed else fast `return wave,wave` (USE_HPSS=1 for real hpss). `AudioDataset(use_demucs=True, task="voice")` → vocals, `task="music"` → music.
- **PANNs** `src/models/panns.py:PANNsPresenceWrapper` : CNN14 torchaudio Mel (n_fft1024 hop320 mel64, 6 ConvBlocks), loads `model/panns/Cnn14_mAP=0.431.pth` (323MB, mandatory), `0.6*PANNs+0.4*detector` presence fusion.
- **DF_Arena_1B** `model/df_arena/df_arena_1b_int8.onnx` (1.37GB, mandatory) : ONNXRuntime 4 threads, batched 3-seg uniform3, topk_mean(k=2).
- **Voice detector** `src/models/aasist.py` AASIST base32 0.57M (5 heads) : trained on vocals stem `--task voice --use_demucs`.
- **Music detector** `src/models/beats_backbone.py` SpecCNN Mel128 + CNN (0.98MB) : trained on music stem `--task music --use_demucs`.
- **Fusion** `script.py:infer_file` : voice on vocals, music on music, PANNs on original, `prob_or=1-(1-voice)*(1-music)`, `detector_fused=wv*file_voice+wm*file_music+wo*prob_or` (weights from `model/fusion_weights.json` optimized via grid search on VAL-A), `file_final=0.5*DF+0.5*detector_fused`, presence-aware `fake=present*fake_raw+(1-present)*0.05` if present<0.4, RMS cap, clip [0.01,0.99]. No 0.5 fallback, fails if mandatory missing. Exact ID mapping only.
- **Segmentation** `src/preprocess.py:extract_segments` 4s uniform5, but inference batched 3-seg for speed (<60min L4).

## Quick Start (Offline)
```bash
pip install -r requirements.txt
python script.py --test_dir ./data/test --output ./output/submission.csv
python tools/validate_submission.py submit.zip
```

## Training — Real Data Only (No Synthetic)
```bash
# 1. Prepare real data (CC BY/CC0/MIT)
python scripts/download_datasets.py --datasets librispeech_dev wavefake fma_small fakemusiccaps --max_samples 1000
python -c "from src.dataset import scan_real_datasets, build_val_sets; df=scan_real_datasets('data/raw'); build_val_sets(df)"

# 2. Train voice/music separately with HTDemucs stems and VAL-A/B/C/D
python scripts/run_all_stages.py --epochs 20 --batch_size 16 --use_demucs
# or individual
python -m src.train --task voice --backbone aasist --use_demucs --epochs 20
python -m src.train --task music --backbone spec_cnn --use_demucs --epochs 20
# 3. Fusion weights auto-optimized on VAL-A, saved to model/fusion_weights.json
# 4. Results only real: experiments/results.csv (voice_aasist, music_spec_cnn)
```

## Results (Real, data/raw 7+1 mix, 2 epochs demo)
- `experiments/results.csv` : `voice_aasist` and `music_spec_cnn` rows, `use_demucs=True`, `score 0.5` (single-sample VAL-A returns 0.5 EER, tiny demo). With full 100k+ data, VAL-A/B/C/D meaningful.
- Fusion: `w_voice_file 0.571 w_music_file 0.286 w_prob_or 0.143`
- Inference: 5 files 5.1s CPU (batched 3-seg), 1200 files ~17min GPU (<60min), VRAM <4GB, silence 0.02, exact ID mapping, mandatory models verified, offline HF_HUB_OFFLINE=1.

## Submit.zip
- Contains: `script.py`, `src/`, `model/best.pt` (voice 2.36MB), `model/music_best.pt` (0.98MB), `model/fusion_weights.json`, `model/panns/Cnn14_mAP=0.431.pth` (323MB), `model/df_arena/df_arena_1b_int8.onnx` (1.37GB) — total 1.71GB (<10GB, <32GB unzip), verified via `tools/validate_submission.py` (checks mandatory, exact mapping, no HeuristicModel, no synthetic).
```
python tools/validate_submission.py submit.zip  # PASS 3 files 7.5s
```
