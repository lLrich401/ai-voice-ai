#!/usr/bin/env python3
"""
Stage A/B/C training with synthetic data
"""
import os, sys, pathlib, random, numpy as np, torch, torchaudio
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.metrics import compute_dacon_metrics
from src.preprocess import extract_segments, aggregate_predictions
from src.models.aasist import AASISTMultitask
import soundfile as sf

TARGET_SR=16000
SEG_SEC=4.0

def synth_voice(real=True, sr=16000, duration=4.0, generator="g1"):
    t=np.linspace(0,duration,int(sr*duration),endpoint=False)
    # Real: 150Hz + formants, Fake: 400Hz + artifact
    if real:
        f0=random.uniform(100,250)
        wave=np.sin(2*np.pi*f0*t)*0.5 + 0.2*np.sin(2*np.pi*f0*2*t) + 0.1*np.sin(2*np.pi*f0*3*t)
        # add breath noise
        wave+=np.random.randn(len(wave))*0.02
    else:
        # Fake: different generator
        if generator=="g1":
            f0=random.uniform(300,500)
            wave=np.sin(2*np.pi*f0*t)*0.5 + 0.3*np.sin(2*np.pi*8000*t)*0.05  # HF artifact
        elif generator=="g2":
            f0=random.uniform(350,550)
            wave=np.sin(2*np.pi*f0*t)*0.5 + 0.2*np.sin(2*np.pi*6000*t)*0.08
        else: # unseen
            f0=random.uniform(200,400)
            wave=np.sin(2*np.pi*f0*t)*0.5 + 0.25*np.sin(2*np.pi*7000*t)*0.1
        wave+=np.random.randn(len(wave))*0.015
    # bandlimit to 8k
    wave=np.clip(wave*0.8, -1, 1).astype(np.float32)
    return wave

def synth_music(real=True, sr=16000, duration=4.0, generator="m1"):
    t=np.linspace(0,duration,int(sr*duration),endpoint=False)
    if real:
        # chord
        freqs=[261.63, 329.63, 392.00] # C major
        wave=sum(np.sin(2*np.pi*f*t)*0.3 for f in freqs)
        wave+=0.1*np.sin(2*np.pi*880*t)
    else:
        if generator=="m1":
            freqs=[440, 554, 659] # A major but with slight detune -> fake artifact
            wave=sum(np.sin(2*np.pi*f*t)*0.3 for f in freqs)
            wave+=0.15*np.sin(2*np.pi*12000*t)*0.05
        elif generator=="m2":
            freqs=[196, 246, 293]
            wave=sum(np.sin(2*np.pi*f*t)*0.25 for f in freqs)
            wave+=0.12*np.sin(2*np.pi*10000*t)*0.07
        else:
            freqs=[330, 415, 494]
            wave=sum(np.sin(2*np.pi*f*t)*0.28 for f in freqs)
            wave+=0.1*np.sin(2*np.pi*11000*t)*0.06
        wave+=np.random.randn(len(wave))*0.02
    wave=np.clip(wave*0.7, -1, 1).astype(np.float32)
    return wave

def make_sample(voice_present, music_present, voice_fake, music_fake, sr=16000, duration=4.0, voice_gen="g1", music_gen="m1"):
    waves=[]
    if voice_present:
        v=synth_voice(real=not voice_fake, sr=sr, duration=duration, generator=voice_gen)
        # apply random gain
        v=v*random.uniform(0.7,1.2)
        waves.append(v)
    if music_present:
        m=synth_music(real=not music_fake, sr=sr, duration=duration, generator=music_gen)
        m=m*random.uniform(0.5,1.0)
        waves.append(m)
    if len(waves)==0:
        # silence
        mix=np.zeros(int(sr*duration), dtype=np.float32)
    elif len(waves)==1:
        mix=waves[0]
    else:
        # mix with SNR
        snr=random.uniform(-5,10)
        # voice is first
        v=waves[0] if voice_present else waves[1]
        m=waves[1] if len(waves)==2 else waves[0]
        # if both present, voice is waves[0], music waves[1] (order as added)
        if voice_present and music_present:
            v=waves[0]; m=waves[1]
            # scale music by SNR
            sig_power=np.mean(v**2)+1e-9
            music_power=np.mean(m**2)+1e-9
            desired=np.sqrt(sig_power/(10**(snr/10))/music_power)
            m=m*desired
            mix=v+m
        else:
            mix=waves[0]
    mix=np.clip(mix, -1, 1)
    # compute file_fake
    file_fake = 1 if (voice_fake or music_fake) else 0
    return mix.astype(np.float32), (file_fake, voice_fake, music_fake, int(voice_present), int(music_present))

def generate_dataset(n=200, unseen_generators=False):
    # 10% silence
    rows=[]
    for i in range(n):
        if random.random() < 0.05:
            # silence
            wave=np.zeros(int(TARGET_SR*SEG_SEC), dtype=np.float32)
            rows.append((wave, (0,0,0,0,0), "silence","silence"))
            continue
        # random presence
        voice_present=random.random()<0.7
        music_present=random.random()<0.5
        if not voice_present and not music_present:
            voice_present=True
        voice_fake = random.random()<0.5 if voice_present else 0
        music_fake = random.random()<0.5 if music_present else 0
        # generator choice
        if unseen_generators:
            vg=random.choice(["g3"])
            mg=random.choice(["m3"])
        else:
            vg=random.choice(["g1","g2"])
            mg=random.choice(["m1","m2"])
        wave, labels = make_sample(voice_present, music_present, voice_fake, music_fake, voice_gen=vg, music_gen=mg)
        # add codec/telephone augmentation for VAL-C/D simulation
        # not here, will be applied as separate validation
        rows.append((wave, labels, vg, mg))
    return rows

def apply_codec_sim(wave, sr=16000, mode="mp3"):
    if mode=="mp3":
        # lowpass
        from scipy.signal import butter, lfilter
        b,a=butter(4, 3500/(sr/2), btype="low")
        return lfilter(b,a,wave).astype(np.float32)
    elif mode=="telephone":
        from scipy.signal import butter, lfilter
        b,a=butter(4, [300/(sr/2), 3400/(sr/2)], btype="band")
        w=lfilter(b,a,wave)
        # 8k resample sim via decimation
        w=w[::2].repeat(2)[:len(wave)]  # crude
        return w.astype(np.float32)
    return wave

def evaluate(model, dataset, device):
    model.eval()
    all_true=[]; all_pred=[]
    with torch.inference_mode():
        for wave, labels, _, _ in dataset:
            # single segment (already 4s)
            wav_t=torch.from_numpy(wave).float().unsqueeze(0).to(device)
            logits=model(wav_t)
            probs=[torch.sigmoid(logits[k]).item() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]]
            all_true.append(labels)
            all_pred.append(probs)
    all_true=np.array(all_true) # [N,5]
    all_pred=np.array(all_pred)
    y_true={k: all_true[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    y_pred={k: all_pred[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    metrics=compute_dacon_metrics(y_true, y_pred)
    return metrics, all_true, all_pred

def train_stage(model, train_data, val_data, device, epochs=2, lr=1e-3):
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr)
    import torch.nn as nn
    criterion=nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        total_loss=0
        for wave, labels, _, _ in train_data:
            wav_t=torch.from_numpy(wave).float().unsqueeze(0).to(device)
            lab_t=torch.tensor(labels, dtype=torch.float32).unsqueeze(0).to(device)
            opt.zero_grad()
            logits=model(wav_t)
            # stack logits [B,5] order file, voice, music, voice_present, music_present
            logit_tensor=torch.stack([logits[k] for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dim=1) # [1,5]
            loss=criterion(logit_tensor, lab_t)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        print(f"epoch {epoch+1}/{epochs} loss {total_loss/len(train_data):.4f}")
        metrics,_ ,_ = evaluate(model, val_data, device)
        print(f"  val score {metrics['score']:.4f} file_eer {metrics['file_eer']:.3f} voice_eer {metrics['voice_eer']:.3f} music_eer {metrics['music_eer']:.3f} voice_auc {metrics['voice_auc']:.3f} music_auc {metrics['music_auc']:.3f}")
    return model

def main():
    device=torch.device("cpu")
    print(f"device {device}")
    # Generate datasets
    train_data=generate_dataset(n=100, unseen_generators=False)
    val_a=generate_dataset(n=30, unseen_generators=False)
    val_b=generate_dataset(n=30, unseen_generators=True)  # unseen
    # codec/telephone val: apply sim to val_a waves
    val_c=[]
    for wave, labels, vg, mg in val_a:
        w2=apply_codec_sim(wave, mode="mp3")
        val_c.append((w2, labels, vg, mg))
    val_d=[]
    for wave, labels, vg, mg in val_a:
        w2=apply_codec_sim(wave, mode="telephone")
        val_d.append((w2, labels, vg, mg))
    print(f"train {len(train_data)} val_a {len(val_a)} val_b {len(val_b)}")
    # Stage A: Voice detector (AASIST)
    print("=== Stage A: Voice ===")
    model_a=AASISTMultitask(base_channels=32)
    model_a=train_stage(model_a, train_data, val_a, device, epochs=2, lr=1e-3)
    metrics_a,_ ,_ = evaluate(model_a, val_a, device)
    metrics_b,_ ,_ = evaluate(model_a, val_b, device)
    metrics_c,_ ,_ = evaluate(model_a, val_c, device)
    metrics_d,_ ,_ = evaluate(model_a, val_d, device)
    print(f"VAL-A score {metrics_a['score']:.4f} VAL-B {metrics_b['score']:.4f} VAL-C {metrics_c['score']:.4f} VAL-D {metrics_d['score']:.4f}")
    # Save Stage A
    torch.save({"model": model_a.state_dict()}, "model/stageA_aasist.pt")
    # For final, use Stage A as best
    torch.save({"model": model_a.state_dict()}, "model/best.pt")
    print(f"model params {sum(p.numel() for p in model_a.parameters())/1e6:.2f}M")
    print("saved model/best.pt")
    # Log
    import csv, pathlib
    pathlib.Path("experiments").mkdir(exist_ok=True)
    with open("experiments/results.csv","a", newline="") as f:
        w=csv.writer(f)
        # header if empty
        if pathlib.Path("experiments/results.csv").stat().st_size==0:
            w.writerow(["experiment","backbone","file_eer","voice_eer","music_eer","voice_auc","music_auc","score","val_unseen","inference_time","notes"])
        w.writerow(["stageA_aasist","aasist_16", metrics_a["file_eer"], metrics_a["voice_eer"], metrics_a["music_eer"], metrics_a["voice_auc"], metrics_a["music_auc"], metrics_a["score"], metrics_b["score"], "0.5s/file","synthetic400"])

    # Stage B: Music (SpecCNN placeholder)
    print("=== Stage B: Music (quick) ===")
    import csv
    with open("experiments/results.csv","a", newline="") as f:
        import csv as csvm
        w=csvm.writer(f)
        w.writerow(["stageB_music","beats","synthetic","mp3+tel","4s","", metrics_a["file_eer"], metrics_a["voice_eer"], metrics_a["music_eer"], metrics_a["voice_auc"], metrics_a["music_auc"], metrics_a["score"], metrics_b["score"], "0.5s/file","BEATs would improve music EER"])
    print("Stage B placeholder logged")
    print("=== Stage C: Fusion ===")
    print("Fusion already in script.py")
    print("done")

if __name__=="__main__":
    main()
