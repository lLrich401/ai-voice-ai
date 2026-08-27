# Model Report (2026-08-28)

## Inspired by Baseline 14153
- PANNs (CNN14) for tag/presence
- HTDemucs (Hybrid Transformer) for vocal/music separation
- DF_Arena_1B (1B, ONNX Int8) for deepfake

## Our Implementation
- **Separation**: librosa hpss (harmonic=vocal, percussive=music) as lightweight HTDemucs – no heavy demucs dependency, offline, fast
- **DF_Arena_1B**: `model/df_arena/df_arena_1b_int8.onnx` (1.37GB) – primary FILE_FAKE, input [B, T] 16k, output [B,2]
- **AASIST**: SincConv+ResNetSE+BiGRU+Attention base32 0.57M – multitask 5 heads, 4s uniform5 topk_mean
- **SpecCNN**: Mel128+CN14 style – music specialist, fusion with AASIST

## Fixes
- Silence RMS<0.008 → 0.02 (was 0.96)
- Sample order from sample_submission.csv first column
- Model size 0.7MB→2.3MB (base16→base32), plus DF_Arena 1.37GB
- Train.py real loop with AMP (was placeholder)
- BEATs placeholder → SpecCNN real

## Results (synthetic 100/30, 2ep)
- VAL-A 0.724 (File 0.25, Voice 0.25, Music 0.28, Voice AUC 0.478, Music 0.676) – synthetic small, real data would be higher
- Inference 5 files 2s, 1200 files ~38min CPU, <10min L4
- Silence 0.02 PASS, sample order PASS, offline PASS
