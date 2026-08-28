# Data Sources

## Real Voice
- LibriSpeech CC BY 4.0 https://www.openslr.org/12
- VoxCeleb2 CC BY 4.0
- Common Voice CC-0

## Fake Voice
- ASVspoof 2019/2021/5
- WaveFake MIT
- MLAAD CC BY 4.0
- DF_Arena training data (not used for training, only inference)

## Real Music
- FMA CC BY 4.0
- MusicCaps CC BY 4.0

## Fake Music
- FakeMusicCaps CC BY 4.0
- MusicGen MIT

## Models (Pretrained, not training data)
- DF_Arena_1B https://huggingface.co/pranjal-pravesh/df_arena_1b (MIT, 1.37GB ONNX Int8) – inference only
- PANNs (CNN14) https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT) – inspiration for SpecCNN
- HTDemucs https://github.com/facebookresearch/demucs (MIT) – inspiration for hpss separation

All training data licenses CC BY/CC0/MIT, research allowed.
