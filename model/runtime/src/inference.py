"""
Unified inference aligned with script.py: PANNs + HTDemucs + DF_Arena + AASIST
Supports silence, topk aggregation, presence calibration, DF_Arena fusion.
"""
from .preprocess import load_audio, extract_segments, aggregate_predictions
import torch, numpy as np, pathlib

def is_silence(wave, thresh=0.008):
    return float((wave**2).mean()**0.5) < thresh

def infer_file(aasist_model, path, device="cpu", df_sess=None, panns_model=None, sr=16000):
    wave,_=load_audio(path)
    if is_silence(wave):
        return [0.05, 0.05, 0.05, 0.02, 0.02]
    # separation for RMS check (optional)
    try:
        from .models.demucs_wrapper import get_separator
        sep = get_separator(device=device)
        wav_vocals, wav_music = sep.separate(wave, sr=sr)
    except:
        wav_vocals, wav_music = wave, wave

    # DF_Arena
    file_fake_df = 0.5
    has_df = df_sess is not None
    if has_df:
        try:
            # reuse df_arena_predict logic
            segs = extract_segments(wave, sr=sr, seg_sec=4.0)
            probs=[]
            inp_name = df_sess.get_inputs()[0].name
            for seg in segs:
                inp = seg[np.newaxis,:].astype(np.float32)
                logits = df_sess.run(None, {inp_name: inp})[0]
                if logits.shape[-1]==1:
                    prob = float(torch.sigmoid(torch.tensor(logits[0,0])).item())
                else:
                    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    prob = exp / np.sum(exp, axis=1, keepdims=True)
                    prob = float(prob[0,1])
                probs.append(prob)
            # topk mean 2
            file_fake_df = float(np.mean(np.sort(probs)[-min(2,len(probs)):]))
            file_fake_df = float(np.clip(file_fake_df,0.01,0.99))
        except:
            has_df=False
            file_fake_df=0.5

    # AASIST
    segs = extract_segments(wave, sr=sr, seg_sec=4.0)
    all_probs=[]
    with torch.inference_mode():
        for seg in segs:
            batch=torch.from_numpy(seg).float().unsqueeze(0).to(device)
            if device=="cuda":
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits=aasist_model(batch)
                except:
                    logits=aasist_model(batch)
            else:
                logits=aasist_model(batch)
            if isinstance(logits, dict):
                probs=[torch.sigmoid(logits[k]).item() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]]
            else:
                probs=torch.sigmoid(logits).cpu().numpy()[0].tolist()
                probs=(probs+[0.5]*5)[:5]
            all_probs.append(probs)
    all_probs=np.array(all_probs)
    # aggregates
    def topk(arr):
        k=min(2,len(arr))
        return float(np.mean(np.sort(arr)[-k:]))
    file_fake_aasist=topk(all_probs[:,0])
    voice_fake_raw=topk(all_probs[:,1])
    music_fake_raw=topk(all_probs[:,2])
    voice_present_aasist=float(np.mean(all_probs[:,3]))
    music_present_aasist=float(np.mean(all_probs[:,4]))

    # PANNs presence
    voice_present_panns=None
    music_present_panns=None
    if panns_model is not None:
        try:
            with torch.inference_mode():
                panns_seg_v=[]; panns_seg_m=[]
                for seg in segs[:3]:
                    batch=torch.from_numpy(seg).float().unsqueeze(0).to(device)
                    out=panns_model(batch)
                    panns_seg_v.append(float(out["voice_present"].item()))
                    panns_seg_m.append(float(out["music_present"].item()))
                voice_present_panns=float(np.mean(panns_seg_v))
                music_present_panns=float(np.mean(panns_seg_m))
        except:
            pass

    if voice_present_panns is not None and getattr(panns_model,'pretrained_loaded',False):
        voice_present=0.6*voice_present_panns+0.4*voice_present_aasist
        music_present=0.6*music_present_panns+0.4*music_present_aasist
    elif voice_present_panns is not None:
        voice_present=voice_present_aasist
        music_present=music_present_aasist
    else:
        voice_present=voice_present_aasist
        music_present=music_present_aasist

    # presence-aware fake
    if voice_present<0.4:
        voice_fake=voice_present*voice_fake_raw+(1-voice_present)*0.05
        voice_fake=float(np.clip(0.3*voice_fake_raw+0.7*voice_fake,0.01,0.99))
    else:
        voice_fake=float(voice_fake_raw)
    if music_present<0.4:
        music_fake=music_present*music_fake_raw+(1-music_present)*0.05
        music_fake=float(np.clip(0.3*music_fake_raw+0.7*music_fake,0.01,0.99))
    else:
        music_fake=float(music_fake_raw)

    # RMS check
    try:
        rms_orig=float((wave**2).mean()**0.5+1e-9)
        rms_voc=float((wav_vocals**2).mean()**0.5+1e-9)
        rms_mus=float((wav_music**2).mean()**0.5+1e-9)
        if rms_voc <0.15*rms_orig:
            voice_present=min(voice_present,0.35)
        if rms_mus <0.15*rms_orig:
            music_present=min(music_present,0.35)
    except:
        pass

    if has_df:
        file_fused=0.5*file_fake_df+0.5*file_fake_aasist
        p_or=1-(1-voice_fake)*(1-music_fake)
        if voice_present<0.3 and music_present<0.3:
            file_final=0.6*file_fused+0.4*file_fake_aasist
        else:
            file_final=0.4*file_fused+0.3*p_or+0.3*file_fake_aasist
        file_final=float(np.clip(file_final,0.01,0.99))
    else:
        p_or=1-(1-voice_fake)*(1-music_fake)
        file_final=0.6*file_fake_aasist+0.4*p_or
        file_final=float(np.clip(file_final,0.01,0.99))

    return [float(np.clip(file_final,0.01,0.99)), float(np.clip(voice_fake,0.01,0.99)), float(np.clip(music_fake,0.01,0.99)), float(np.clip(voice_present,0.01,0.99)), float(np.clip(music_present,0.01,0.99))]
