# SSL backbone placeholder - would use transformers WavLM
import torch, torch.nn as nn
class SSLMultitask(nn.Module):
    def __init__(self): super().__init__(); self.dummy=nn.Linear(10,5)
    def forward(self, x): return {"file_fake":self.dummy(torch.randn(x.size(0),10))[:,0]}
