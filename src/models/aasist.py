import torch
import torch.nn as nn
import torch.nn.functional as F

class SincConv(nn.Module):
    def __init__(self, out_channels=20, kernel_size=1024, in_channels=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.3)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.act = nn.LeakyReLU(0.3)
        self.se = SqueezeExcitation(out_channels)
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.act(out)

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x * y.expand_as(x)

class AASISTBackbone(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.sinc = SincConv(out_channels=20, kernel_size=1024)
        self.maxpool = nn.MaxPool1d(3, stride=3)
        self.layer1 = self._make_layer(20, base_channels, blocks=2, stride=2)
        self.layer2 = self._make_layer(base_channels, base_channels*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*2, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels*2, base_channels*4, blocks=2, stride=2)
        self.gru = nn.GRU(base_channels*4, base_channels*4, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(base_channels*8, base_channels*4),
            nn.Tanh(),
            nn.Linear(base_channels*4, 1)
        )
        self.out_dim = base_channels*8
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride !=1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.sinc(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(1,2)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        attn_weights = self.attention(out)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(out * attn_weights, dim=1)
        return pooled

class AASISTMultitask(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.backbone = AASISTBackbone(base_channels=base_channels)
        dim = self.backbone.out_dim
        self.voice_fake_head = nn.Linear(dim, 1)
        self.music_fake_head = nn.Linear(dim, 1)
        self.voice_present_head = nn.Linear(dim, 1)
        self.music_present_head = nn.Linear(dim, 1)
        self.file_fake_head = nn.Linear(dim, 1)
    def forward(self, wav):
        if wav.dim()==2:
            wav = wav.unsqueeze(1)
        feat = self.backbone(wav)
        logits = {
            "voice_fake": self.voice_fake_head(feat).squeeze(-1),
            "music_fake": self.music_fake_head(feat).squeeze(-1),
            "voice_present": self.voice_present_head(feat).squeeze(-1),
            "music_present": self.music_present_head(feat).squeeze(-1),
            "file_fake": self.file_fake_head(feat).squeeze(-1),
        }
        return logits
