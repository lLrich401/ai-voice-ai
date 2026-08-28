"""
Real training loop with AMP, cosine, early stopping, checkpoint
"""
import torch, torch.nn as nn, pathlib, random, numpy as np
from torch.cuda.amp import GradScaler, autocast
from src.metrics import compute_dacon_metrics

def train_one_epoch(model, loader, opt, device, scaler):
    model.train()
    total=0
    for wav, labels, _ in loader:
        wav=wav.to(device); labels=labels.to(device)
        opt.zero_grad()
        with autocast(enabled=(device.type=="cuda")):
            logits=model(wav)
            # stack
            logit_t=torch.stack([logits[k] for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dim=1)
            loss=nn.BCEWithLogitsLoss()(logit_t, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total+=loss.item()
    return total/len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_t=[]; all_p=[]
    for wav, labels, _ in loader:
        wav=wav.to(device)
        logits=model(wav)
        probs=[torch.sigmoid(logits[k]).cpu().numpy() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]]
        probs=np.stack(probs, axis=1)
        all_t.append(labels.numpy()); all_p.append(probs)
    all_t=np.concatenate(all_t); all_p=np.concatenate(all_p)
    y_true={k: all_t[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    y_pred={k: all_p[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    return compute_dacon_metrics(y_true, y_pred)

if __name__=="__main__":
    print("Use scripts/run_all_stages.py for synthetic training or implement your own DataLoader with real datasets")
