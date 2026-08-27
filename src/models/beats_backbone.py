"""
BEATs-like General Audio / Music forensics backbone
Uses MelSpectrogram + Efficient CNN (PANNs/CNN14 style)
Not a full BEATs transformer, but a lightweight CNN that is trainable and effective for music fake detection.
For real BEATs, would load facebook/beats via transformers, but this CNN is a practical substitute for offline and fast inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SpecCNNBackbone(nn.Module):
    def __init__(self, n_mels=128, base_channels=32):
        super().__init__()
        self.n_mels=n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, win_length=400, hop_length=160,
            n_mels=n_mels, f_min=0, f_max=8000, power=2.0
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1), nn.BatchNorm2d(base_channels*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.out_dim = base_channels*4
        self.proj = nn.Sequential(nn.Linear(self.out_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.out_dim = 256

    def forward(self, wav):
        # wav: [B,T] 16k
        # Compute mel
        # wav is [B,T], need [B,T] for mel
        mel = self.mel(wav)  # [B, n_mels, T]
        mel = self.db(mel)  # log
        mel = (mel + 80)/80  # norm 0-1
        mel = mel.unsqueeze(1)  # [B,1,n_mels,T]
        feat = self.cnn(mel)  # [B,C,1,1]
        feat = feat.flatten(1)  # [B,C]
        feat = self.proj(feat)
        return feat

class MusicMultitask(nn.Module):
    def __init__(self, n_mels=128, base_channels=32):
        super().__init__()
        self.backbone = SpecCNNBackbone(n_mels=n_mels, base_channels=base_channels)
        dim = self.backbone.out_dim
        self.heads = nn.ModuleDict({
            "file_fake": nn.Linear(dim,1),
            "voice_fake": nn.Linear(dim,1),
            "music_fake": nn.Linear(dim,1),
            "voice_present": nn.Linear(dim,1),
            "music_present": nn.Linear(dim,1),
        })
    def forward(self, wav):
        feat = self.backbone(wav)
        return {k: self.heads[k](feat).squeeze(-1) for k in self.heads}

class FusionModel(nn.Module):
    def __init__(self, aasist_channels=32, music_channels=32):
        super().__init__()
        from .aasist import AASISTBackbone
        self.aasist = AASISTBackbone(base_channels=aasist_channels)
        self.music = SpecCNNBackbone(base_channels=music_channels)
        total = self.aasist.out_dim + self.music.out_dim
        self.fusion = nn.Sequential(nn.Linear(total, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,128), nn.ReLU())
        dim=128
        self.heads = nn.ModuleDict({k: nn.Linear(dim,1) for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]})
    def forward(self, wav):
        f1 = self.aasist(wav.unsqueeze(1) if wav.dim()==2 else wav)
        f2 = self.music(wav)
        fused = torch.cat([f1,f2], dim=1)
        fused = self.fusion(fused)
        return {k: self.heads[k](fused).squeeze(-1) for k in self.heads}
