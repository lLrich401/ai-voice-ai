# Multitask registry placeholder
MODEL_REGISTRY={}
def create_model(name, **kw): return None
class EnsembleModel(torch.nn.Module):
    def __init__(self, models): super().__init__(); self.models=models
    def forward(self, x): return self.models[0](x)
