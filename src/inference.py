from .preprocess import load_audio, extract_segments, aggregate_predictions
import torch, numpy as np
def infer_file(model, path, device):
    wave,_=load_audio(path)
    segs=extract_segments(wave)
    probs=[]
    with torch.inference_mode():
        for seg in segs:
            batch=torch.from_numpy(seg).float().unsqueeze(0).to(device)
            logits=model(batch)
            probs.append([torch.sigmoid(logits[k]).item() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]])
    probs=np.array(probs)
    return [float(np.mean(probs[:,i])) for i in range(5)]
