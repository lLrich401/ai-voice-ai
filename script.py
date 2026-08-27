#!/usr/bin/env python3
import os, sys, pathlib, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
import numpy as np, pandas as pd, torch, torch.nn as nn
import soundfile as sf
TARGET_SR=16000

# Try onnxruntime for DF_Arena
try:
    import onnxruntime as ort
    HAS_ONNX=True
except:
    HAS_ONNX=False
    ort=None

# Try demucs, fallback to librosa hpss
try:
    import demucs
    HAS_DEMUCS=True
except:
    HAS_DEMUCS=False

def load_audio(path, target_sr=TARGET_SR):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim==2:
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

# Load AASIST (from src)
try:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from src.models.aasist import AASISTMultitask
    HAS_AASIST=True
except:
    HAS_AASIST=False
    AASISTMultitask=None

class HeuristicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(10,32), nn.ReLU(), nn.Linear(32,5))
        with torch.no_grad():
            for p in self.head.parameters():
                p.zero_()
    def forward(self, wav):
        feats = torch.randn(wav.size(0),10, device=wav.device)
        logits = self.head(feats)
        return {"file_fake":logits[:,0],"voice_fake":logits[:,1],"music_fake":logits[:,2],"voice_present":logits[:,3],"music_present":logits[:,4]}

def separate_vocals_music(wave, sr=16000):
    # Try HTDemucs if available, else librosa hpss
    if HAS_DEMUCS:
        try:
            # Placeholder: would use demucs.api.Separator
            # For now fallback to hpss
            pass
        except:
            pass
    try:
        import librosa
        # hpss returns harmonic and percussive
        # Use harmonic as vocals, percussive as music
        # Need to do stft
        y_harm, y_perc = librosa.effects.hpss(wave)
        # y_harm is harmonic (vocals), y_perc is percussive (music)
        # Ensure same length
        if len(y_harm) != len(wave):
            y_harm = np.pad(y_harm, (0, max(0, len(wave)-len(y_harm))))[:len(wave)]
            y_perc = np.pad(y_perc, (0, max(0, len(wave)-len(y_perc))))[:len(wave)]
        return y_harm.astype(np.float32), y_perc.astype(np.float32)
    except:
        # Fallback: return original for both
        return wave, wave

def load_df_arena(device="cpu"):
    # Try to load DF_Arena ONNX
    if not HAS_ONNX:
        return None
    model_path = pathlib.Path("model/df_arena/df_arena_1b_int8.onnx")
    if not model_path.exists():
        model_path = pathlib.Path(__file__).parent / "model" / "df_arena" / "df_arena_1b_int8.onnx"
    if not model_path.exists():
        return None
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(model_path), providers=providers)
        print(f"loaded DF_Arena from {model_path} providers {providers}")
        return sess
    except Exception as e:
        print(f"DF_Arena load failed {e}")
        return None

def df_arena_predict(sess, wave, sr=16000, seg_sec=4.0):
    # Run DF_Arena on segments, aggregate
    segs = extract_segments(wave, sr=sr, seg_sec=seg_sec)
    probs = []
    for seg in segs:
        # seg is [64000], need [1, 64000]
        inp = seg[np.newaxis, :].astype(np.float32)
        logits = sess.run(None, {"input_values": inp})[0]  # [1,2]
        # softmax
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        # prob[:,1] is fake
        probs.append(prob[0,1])
    # aggregate topk
    return aggregate_predictions(probs, method="topk_mean", top_k=2)

def load_aasist(device):
    import pathlib as pl
    ckpt_path = pl.Path("model/best.pt")
    if not ckpt_path.exists():
        ckpt_path = pathlib.Path(__file__).parent / "model" / "best.pt"
    if ckpt_path.exists() and HAS_AASIST:
        for base_ch in [32, 16, 64]:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                sd = ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
                model = AASISTMultitask(base_channels=base_ch)
                missing, _ = model.load_state_dict(sd, strict=False)
                if len(missing) < 10:
                    model.to(device).eval()
                    print(f"loaded AASIST base{base_ch} from {ckpt_path}")
                    return model
            except Exception as e:
                continue
    print("using HeuristicModel")
    model = HeuristicModel().to(device).eval()
    return model

def infer_file(aasist_model, df_sess, audio_path, device):
    try:
        wave, sr = load_audio(audio_path)
    except:
        return [0.5]*5
    if is_silence(wave, thresh=0.008):
        return [0.05, 0.05, 0.05, 0.02, 0.02]
    # Separate
    wav_vocals, wav_music = separate_vocals_music(wave, sr=sr)
    # DF_Arena for file fake on original
    file_fake = 0.5
    if df_sess is not None:
        try:
            file_fake = df_arena_predict(df_sess, wave, sr=sr)
            file_fake = float(np.clip(file_fake, 0.01, 0.99))
        except:
            file_fake = 0.5
    # AASIST for voice/music
    # Use AASIST on vocals for voice, on music for music, and on original for presence
    # For simplicity, run AASIST on original and use its heads
    segs = extract_segments(wave, sr=sr)
    # Also get segs for vocals/music if needed
    all_probs = []
    with torch.inference_mode():
        for seg in segs:
            batch = torch.from_numpy(seg).float().unsqueeze(0).to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = aasist_model(batch)
            else:
                logits = aasist_model(batch)
            if isinstance(logits, dict):
                probs = [torch.sigmoid(logits[k]).item() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]]
            else:
                probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()
            all_probs.append(probs)
    all_probs = np.array(all_probs)
    # Aggregate
    voice_fake = aggregate_predictions(all_probs[:,1], method="topk_mean", top_k=2)
    music_fake = aggregate_predictions(all_probs[:,2], method="topk_mean", top_k=2)
    voice_present = float(np.mean(all_probs[:,3]))
    music_present = float(np.mean(all_probs[:,4]))
    # File fake from AASIST direct
    file_fake_aasist = aggregate_predictions(all_probs[:,0], method="topk_mean", top_k=2)
    # Fuse with DF_Arena: weighted 0.5 DF + 0.5 AASIST, plus probabilistic
    if df_sess is not None:
        # file_fake is from DF_Arena, blend with AASIST
        file_fused = 0.5*file_fake + 0.5*file_fake_aasist
        # Also probabilistic OR
        p_or = 1 - (1-voice_fake)*(1-music_fake)
        file_final = 0.4*file_fused + 0.3*p_or + 0.3*file_fake_aasist
        file_final = float(np.clip(file_final, 0.01, 0.99))
    else:
        # No DF, use probabilistic
        p_or = 1 - (1-voice_fake)*(1-music_fake)
        file_final = 0.6*file_fake_aasist + 0.4*p_or
        file_final = float(np.clip(file_final, 0.01, 0.99))
    # Clamp
    return [file_final, float(np.clip(voice_fake,0.01,0.99)), float(np.clip(music_fake,0.01,0.99)), float(np.clip(voice_present,0.01,0.99)), float(np.clip(music_present,0.01,0.99))]

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="./data/test")
    parser.add_argument("--output", default="./output/submission.csv")
    args=parser.parse_args()
    test_dir=args.test_dir
    if not os.path.exists(test_dir):
        test_dir="./data/test"
    test_path=pathlib.Path(test_dir)
    audio_files=sorted(test_path.rglob("*.wav"))+sorted(test_path.rglob("*.mp3"))+sorted(test_path.rglob("*.flac"))
    audio_files=sorted(set(audio_files))
    print(f"found {len(audio_files)} files")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    aasist_model=load_aasist(device)
    df_sess=load_df_arena(device)
    # warmup
    try:
        dummy=torch.randn(1,16000*4).to(device)
        with torch.inference_mode():
            _=aasist_model(dummy)
        if df_sess is not None:
            dummy_np=np.random.randn(1,16000*4).astype(np.float32)
            _=df_sess.run(None, {"input_values": dummy_np})
        print("warmup ok")
    except Exception as e:
        print(f"warmup fail {e}")
    # Handle sample_submission
    sample_path=None
    for cand in ["./sample_submission.csv","sample_submission.csv","./data/sample_submission.csv", str(pathlib.Path(__file__).parent / "sample_submission.csv")]:
        if os.path.exists(cand):
            sample_path=cand
            break
    results=[]
    for af in audio_files:
        probs=infer_file(aasist_model, df_sess, str(af), device)
        # Use stem as id, but will be reordered if sample exists
        results.append([af.stem]+probs)
    # Reorder if sample exists
    if sample_path:
        try:
            sdf=pd.read_csv(sample_path)
            sample_ids=sdf.iloc[:,0].astype(str).tolist()
            id_to_probs={str(r[0]): r[1:] for r in results}
            ordered=[]
            for sid in sample_ids:
                if sid in id_to_probs:
                    ordered.append([sid]+id_to_probs[sid])
                else:
                    # try stem match
                    found=False
                    for k,v in id_to_probs.items():
                        if sid==k or sid in k or k in sid:
                            ordered.append([sid]+v); found=True; break
                    if not found:
                        ordered.append([sid,0.5,0.5,0.5,0.5,0.5])
            df=pd.DataFrame(ordered,columns=[sdf.columns[0],"FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
        except Exception as e:
            print(f"sample handling failed {e}")
            df=pd.DataFrame(results,columns=["id","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
    else:
        df=pd.DataFrame(results,columns=["id","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output,index=False)
    print(f"saved {args.output} {len(df)} rows")
    alt=pathlib.Path("./output/submission.csv")
    if str(alt)!=args.output:
        alt.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(alt,index=False)

if __name__=="__main__":
    main()
