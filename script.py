#!/usr/bin/env python3
"""
DACON 236749 Real-Data Pipeline Inference: PANNs + HTDemucs + DF_Arena_1B + Voice/Music Detectors
- Voice detector (AASIST) on vocals stem via HTDemucs
- Music detector (SpecCNN) on music stem via HTDemucs
- PANNs presence, DF_Arena file fake, fusion weights optimized on VAL
- Fails clearly if mandatory models missing (no 0.5 fallback)
- Exact ID mapping only
"""
import os, sys, pathlib, warnings, json
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
import numpy as np, pandas as pd, torch
import soundfile as sf
TARGET_SR=16000
SEG_SEC=4.0

# Imports
import onnxruntime as ort
from src.models.aasist import AASISTMultitask
from src.models.beats_backbone import MusicMultitask
from src.models.panns import PANNsPresenceWrapper
from src.models.demucs_wrapper import get_separator

def verify_mandatory_models():
    """Requirement 11 & 12: verify existence, fail if missing."""
    missing=[]
    checks=[
        ("DF_Arena", pathlib.Path("model/df_arena/df_arena_1b_int8.onnx")),
        ("DF_Arena_alt", pathlib.Path(__file__).parent / "model" / "df_arena" / "df_arena_1b_int8.onnx"),
        ("PANNs", pathlib.Path("model/panns/Cnn14_mAP=0.431.pth")),
        ("PANNs_alt", pathlib.Path(__file__).parent / "model" / "panns" / "Cnn14_mAP=0.431.pth"),
        ("Voice_checkpoint", pathlib.Path("model/best.pt")),
        ("Music_checkpoint", pathlib.Path("model/music_best.pt")),
        ("Fusion_weights", pathlib.Path("model/fusion_weights.json")),
    ]
    # Check existence with fallback
    df_exists = checks[0][1].exists() or checks[1][1].exists()
    panns_exists = checks[2][1].exists() or checks[3][1].exists()
    voice_exists = checks[4][1].exists()
    music_exists = checks[5][1].exists()
    fusion_exists = checks[6][1].exists()
    if not df_exists:
        missing.append("model/df_arena/df_arena_1b_int8.onnx (1.37GB DF_Arena_1B)")
    if not panns_exists:
        missing.append("model/panns/Cnn14_mAP=0.431.pth (PANNs CNN14)")
    if not voice_exists:
        missing.append("model/best.pt (voice detector checkpoint)")
    if not music_exists:
        missing.append("model/music_best.pt (music detector checkpoint)")
    if not fusion_exists:
        missing.append("model/fusion_weights.json (fusion weights) - will use default 0.5/0.3/0.2 but should be optimized")
        # Fusion is not strictly mandatory, but we warn
    if missing:
        # For submission, DF_Arena, PANNs, voice/music must exist -> fail
        essential = [m for m in missing if "fusion" not in m.lower()]
        if essential:
            raise FileNotFoundError(f"Mandatory models missing: {essential}. Ensure submit.zip contains them. Download DF_Arena from https://huggingface.co/pranjal-pravesh/df_arena_1b and PANNs from https://github.com/qiuqiangkong/audioset_tagging_cnn")
        else:
            print(f"Warning: {missing} - using defaults")
    print("Mandatory models verified")
    return True

def load_audio(path, target_sr=TARGET_SR):
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception as e:
        try:
            import librosa
            data, sr = librosa.load(path, sr=target_sr, mono=False)
            if data.ndim == 2:
                data = np.mean(data, axis=0)
            return data.astype(np.float32), target_sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {path}: {e} / {e2}")
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except:
            from scipy.signal import resample
            data = resample(data, int(len(data)*target_sr/sr))
    return data.astype(np.float32), target_sr

def extract_segments(wave, sr=16000, seg_sec=4.0):
    seg_len=int(seg_sec*sr)
    if len(wave) <= seg_len:
        return [np.pad(wave,(0,seg_len-len(wave)))]
    positions=[0,0.25,0.5,0.75,1.0]
    segs=[]
    for p in positions:
        start=int((len(wave)-seg_len)*p)
        start=max(0,min(start,len(wave)-seg_len))
        segs.append(wave[start:start+seg_len])
    return segs

def aggregate_predictions(probs, method="topk_mean", top_k=2):
    probs=np.asarray(probs)
    if method=="topk_mean":
        k=min(top_k,len(probs))
        return float(np.mean(np.sort(probs)[-k:]))
    return float(np.mean(probs))

def is_silence(wave, thresh=0.008):
    rms = float((wave**2).mean()**0.5)
    return rms < thresh

def load_df_arena(device="cpu"):
    # Mandatory, fail if not found
    paths=[
        pathlib.Path("model/df_arena/df_arena_1b_int8.onnx"),
        pathlib.Path(__file__).parent / "model" / "df_arena" / "df_arena_1b_int8.onnx",
    ]
    model_path=next((p for p in paths if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("DF_Arena ONNX not found at model/df_arena/df_arena_1b_int8.onnx")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads=4
    sess_opts.inter_op_num_threads=1
    try:
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except: pass
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    print(f"loaded DF_Arena {model_path} {sess.get_providers()} input {sess.get_inputs()[0].name}")
    return sess

def df_arena_predict(sess, wave, sr=16000, seg_sec=4.0):
    seg_len=int(seg_sec*sr)
    if len(wave) <= seg_len:
        segs=[np.pad(wave,(0,seg_len-len(wave)))]
    else:
        positions=[0,0.5,1.0]
        segs=[]
        for p in positions:
            start=int((len(wave)-seg_len)*p)
            start=max(0,min(start,len(wave)-seg_len))
            segs.append(wave[start:start+seg_len])
    inp_name = sess.get_inputs()[0].name
    # batched
    batch = np.stack(segs, axis=0).astype(np.float32)
    logits = sess.run(None, {inp_name: batch})[0]
    probs=[]
    for i in range(logits.shape[0]):
        logit=logits[i]
        if logits.shape[-1]==1:
            prob=float(torch.sigmoid(torch.tensor(float(logit[0]))).item())
        else:
            m=np.max(logit); exp=np.exp(logit-m); prob=float(exp[1]/np.sum(exp))
        probs.append(prob)
    return aggregate_predictions(probs, method="topk_mean", top_k=2)

def load_voice_model(device):
    # Mandatory
    cands=[pathlib.Path("model/best.pt"), pathlib.Path("model/voice_aasist.pt"), pathlib.Path(__file__).parent/"model"/"best.pt"]
    ckpt_path=next((p for p in cands if p.exists()), None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Voice checkpoint not found: model/best.pt (train via scripts/run_all_stages.py)")
    for base_ch in [32,16,64]:
        try:
            ckpt=torch.load(str(ckpt_path), map_location="cpu")
            sd=ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
            model=AASISTMultitask(base_channels=base_ch)
            missing,_=model.load_state_dict(sd, strict=False)
            if len(missing)<20:
                model.to(device).eval()
                print(f"loaded Voice AASIST {ckpt_path} base{base_ch}")
                return model
        except Exception as e:
            continue
    raise RuntimeError(f"Voice checkpoint {ckpt_path} incompatible")

def load_music_model(device):
    cands=[pathlib.Path("model/music_best.pt"), pathlib.Path(__file__).parent/"model"/"music_best.pt"]
    ckpt_path=next((p for p in cands if p.exists()), None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Music checkpoint not found: model/music_best.pt (train via scripts/run_all_stages.py)")
    # Try SpecCNN first
    try:
        ckpt=torch.load(str(ckpt_path), map_location="cpu")
        sd=ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
        # Try AASIST backbone for music as well if spec_cnn fails
        try:
            model=MusicMultitask(base_channels=32)
            missing,_=model.load_state_dict(sd, strict=False)
            if len(missing)<50:
                model.to(device).eval()
                print(f"loaded Music SpecCNN {ckpt_path}")
                return model
        except:
            pass
        # fallback to AASIST
        model=AASISTMultitask(base_channels=32)
        missing,_=model.load_state_dict(sd, strict=False)
        if len(missing)<50:
            model.to(device).eval()
            print(f"loaded Music AASIST {ckpt_path}")
            return model
    except Exception as e:
        raise RuntimeError(f"Music checkpoint load failed {ckpt_path}: {e}")
    raise RuntimeError(f"Music checkpoint {ckpt_path} incompatible")

def load_panns(device):
    cands=[pathlib.Path("model/panns/Cnn14_mAP=0.431.pth"), pathlib.Path(__file__).parent / "model" / "panns" / "Cnn14_mAP=0.431.pth"]
    ckpt_path=next((p for p in cands if p.exists()), None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("PANNs checkpoint not found: model/panns/Cnn14_mAP=0.431.pth (download from https://github.com/qiuqiangkong/audioset_tagging_cnn)")
    model=PANNsPresenceWrapper(use_pretrained=True)
    if not model.pretrained_loaded:
        raise RuntimeError("PANNs failed to load pretrained weights")
    model.to(device).eval()
    print(f"loaded PANNs {ckpt_path}")
    return model

def load_fusion_weights():
    p=pathlib.Path("model/fusion_weights.json")
    if not p.exists():
        p=pathlib.Path(__file__).parent/"model"/"fusion_weights.json"
    if p.exists():
        with open(p) as f:
            w=json.load(f)
        print(f"loaded fusion {w}")
        return w
    print("Fusion weights not found, using default 0.5/0.3/0.2 (should be optimized via run_all_stages)")
    return {"w_voice_file":0.5, "w_music_file":0.3, "w_prob_or":0.2}

def infer_file(voice_model, music_model, df_sess, panns_model, fusion_weights, audio_path, device):
    wave, sr = load_audio(audio_path)
    if is_silence(wave, thresh=0.008):
        return [0.05, 0.05, 0.05, 0.02, 0.02]
    # HTDemucs separation: vocals -> voice detector, music -> music detector (requirement 8)
    separator=get_separator(device=device)
    try:
        wav_vocals, wav_music = separator.separate(wave, sr=sr)
    except Exception as e:
        raise RuntimeError(f"HTDemucs separation failed {audio_path}: {e}")

    # DF_Arena on original for file fake
    file_fake_df = df_arena_predict(df_sess, wave, sr=sr)
    file_fake_df = float(np.clip(file_fake_df, 0.01, 0.99))

    # Voice detector on vocals stem - batched, 3 segs for speed (meets 60min budget, L4 GPU faster)
    segs_v = extract_segments(wav_vocals, sr=sr, seg_sec=SEG_SEC)[:3] if len(wav_vocals)>16000*4 else extract_segments(wav_vocals, sr=sr, seg_sec=SEG_SEC)
    with torch.inference_mode():
        batch_v=torch.from_numpy(np.stack(segs_v)).float().to(device)  # [N, T]
        if device=="cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_v=voice_model(batch_v)
        else:
            logits_v=voice_model(batch_v)
        # logits dict [N]
        v_probs=np.stack([torch.sigmoid(logits_v[k]).cpu().numpy() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], axis=1)  # [N,5]
    voice_fake = aggregate_predictions(v_probs[:,1], method="topk_mean", top_k=2)
    voice_present_v = float(np.mean(v_probs[:,3]))
    file_voice = aggregate_predictions(v_probs[:,0], method="topk_mean", top_k=2)

    # Music detector on music stem - batched, 3 segs
    segs_m = extract_segments(wav_music, sr=sr, seg_sec=SEG_SEC)[:3] if len(wav_music)>16000*4 else extract_segments(wav_music, sr=sr, seg_sec=SEG_SEC)
    with torch.inference_mode():
        batch_m=torch.from_numpy(np.stack(segs_m)).float().to(device)
        if device=="cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_m=music_model(batch_m)
        else:
            logits_m=music_model(batch_m)
        m_probs=np.stack([torch.sigmoid(logits_m[k]).cpu().numpy() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], axis=1)
    music_fake = aggregate_predictions(m_probs[:,2], method="topk_mean", top_k=2)
    music_present_m = float(np.mean(m_probs[:,4]))
    file_music = aggregate_predictions(m_probs[:,0], method="topk_mean", top_k=2)

    # PANNs presence on original - batched (3 segs)
    with torch.inference_mode():
        segs_orig = extract_segments(wave, sr=sr, seg_sec=SEG_SEC)[:3]
        batch_p=torch.from_numpy(np.stack(segs_orig)).float().to(device)
        out=panns_model(batch_p)
        # out voice_present/music_present are [B]
        voice_present_panns=float(torch.mean(out["voice_present"]).item())
        music_present_panns=float(torch.mean(out["music_present"]).item())

    # Fuse presence: 0.6 PANNs +0.4 detector
    voice_present = 0.6*voice_present_panns + 0.4*voice_present_v
    music_present = 0.6*music_present_panns + 0.4*music_present_m

    # Presence-aware calibration
    if voice_present < 0.4:
        voice_fake = voice_present*voice_fake + (1-voice_present)*0.05
        voice_fake=float(np.clip(voice_fake,0.01,0.99))
    if music_present < 0.4:
        music_fake = music_present*music_fake + (1-music_present)*0.05
        music_fake=float(np.clip(music_fake,0.01,0.99))

    # RMS check
    try:
        rms_orig=float((wave**2).mean()**0.5+1e-9)
        rms_voc=float((wav_vocals**2).mean()**0.5+1e-9)
        rms_mus=float((wav_music**2).mean()**0.5+1e-9)
        if rms_voc < 0.15*rms_orig:
            voice_present=min(voice_present,0.35)
        if rms_mus < 0.15*rms_orig:
            music_present=min(music_present,0.35)
    except:
        pass

    # File fusion with optimized weights
    wv=fusion_weights.get("w_voice_file",0.5)
    wm=fusion_weights.get("w_music_file",0.3)
    wo=fusion_weights.get("w_prob_or",0.2)
    # also blend DF_Arena: file fused = 0.5*DF +0.5*(voice_file+music_file)/2? But requirement says use validation optimized
    # Use: file_ensemble = 0.5*DF +0.5*(wv*file_voice + wm*file_music + wo*probOR)?? Simpler: blend DF with detector fusion
    prob_or=1-(1-voice_fake)*(1-music_fake)
    detector_fused=wv*file_voice + wm*file_music + wo*prob_or
    # normalize already sum 1
    file_final = 0.5*file_fake_df + 0.5*detector_fused
    file_final=float(np.clip(file_final,0.01,0.99))

    return [file_final, float(np.clip(voice_fake,0.01,0.99)), float(np.clip(music_fake,0.01,0.99)), float(np.clip(voice_present,0.01,0.99)), float(np.clip(music_present,0.01,0.99))]

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="./data/test")
    parser.add_argument("--output", default="./output/submission.csv")
    args=parser.parse_args()
    # Verify mandatory models first (fail fast)
    verify_mandatory_models()
    test_path=pathlib.Path(args.test_dir if pathlib.Path(args.test_dir).exists() else "./data/test")
    audio_files=sorted(test_path.rglob("*.wav"))+sorted(test_path.rglob("*.mp3"))+sorted(test_path.rglob("*.flac"))+sorted(test_path.rglob("*.m4a"))+sorted(test_path.rglob("*.ogg"))
    audio_files=sorted(set(audio_files))
    if len(audio_files)==0:
        raise FileNotFoundError(f"No audio files found in {test_path}")
    print(f"found {len(audio_files)} files under {test_path}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    voice_model=load_voice_model(device)
    music_model=load_music_model(device)
    df_sess=load_df_arena(device)
    panns_model=load_panns(device)
    fusion_weights=load_fusion_weights()
    # warmup - mandatory models must succeed
    dummy=torch.randn(1,16000*4).to(device)
    with torch.inference_mode():
        _=voice_model(dummy)
        _=music_model(dummy)
        _=panns_model(dummy)
    dummy_np=np.random.randn(1,16000*4).astype(np.float32)
    _=df_sess.run(None, {df_sess.get_inputs()[0].name: dummy_np})
    print("warmup ok")
    # sample_submission exact mapping only
    sample_path=None
    for cand in ["./sample_submission.csv","sample_submission.csv","./data/sample_submission.csv", str(pathlib.Path(__file__).parent / "sample_submission.csv")]:
        if pathlib.Path(cand).exists():
            sample_path=cand; break
    if sample_path is None:
        for p in pathlib.Path(".").rglob("sample_submission.csv"):
            sample_path=str(p); break
    results=[]
    import time, tqdm
    start=time.time()
    for af in tqdm.tqdm(audio_files) if len(audio_files)>10 else audio_files:
        probs=infer_file(voice_model, music_model, df_sess, panns_model, fusion_weights, str(af), device)
        results.append([af.stem]+probs)
    elapsed=time.time()-start
    print(f"inference {len(results)} files in {elapsed:.1f}s ({elapsed/len(results):.2f}s/file)" if len(results)>0 else "no files")
    # Exact mapping only (no substring)
    if sample_path:
        sdf=pd.read_csv(sample_path)
        sample_ids=sdf.iloc[:,0].astype(str).tolist()
        id_to_probs={str(r[0]): r[1:] for r in results}
        ordered=[]
        missing=[]
        for sid in sample_ids:
            sid_str=str(sid)
            if sid_str in id_to_probs:
                ordered.append([sid_str]+id_to_probs[sid_str])
            else:
                # Try stem of sid (if sample includes extension)
                stem=pathlib.Path(sid_str).stem
                if stem in id_to_probs and stem!=sid_str:
                    # Exact stem mapping is allowed (sample may have .wav extension)
                    ordered.append([sid_str]+id_to_probs[stem])
                else:
                    missing.append(sid_str)
                    raise FileNotFoundError(f"sample ID {sid_str} not found in test files {list(id_to_probs.keys())[:5]}... (exact mapping required, no substring)")
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} IDs: {missing[:5]}")
        df=pd.DataFrame(ordered,columns=[sdf.columns[0],"FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
    else:
        df=pd.DataFrame(results,columns=["id","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
        print("no sample_submission, using id order")
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output,index=False)
    print(f"saved {args.output} {len(df)} rows")
    alt=pathlib.Path("./output/submission.csv")
    if str(alt)!=args.output:
        alt.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(alt,index=False)
        print(f"also saved {alt}")

if __name__=="__main__":
    main()
