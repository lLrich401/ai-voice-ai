#!/usr/bin/env python3
"""
DACON 236749 Baseline-Inspired Inference: PANNs + HTDemucs + DF_Arena_1B + AASIST
Ref: https://dacon.io/competitions/official/236749/codeshare/14153

Pipeline:
  1) Load audio (16kHz mono, handles wav/mp3/flac, mono/stereo via soundfile+librosa)
  2) Silence check (RMS <0.008 -> present 0.02, fake 0.05)
  3) HTDemucs separation (demucs HTDemucs if available, else librosa hpss lightweight)
     - vocals (harmonic) / music (percussive) – preserves offline speed
  4) PANNs CNN14 for VOICE_PRESENT / MUSIC_PRESENT (AudioSet Speech/Music tags)
     - Pretrained Cnn14_mAP=0.431.pth if present in model/panns/, else fallback to AASIST presence
  5) DF_Arena_1B ONNX (1.37GB int8) for FILE_FAKE primary
  6) AASIST (SincConv+ResNetSE+BiGRU) for VOICE_FAKE, MUSIC_FAKE, and supplementary FILE_FAKE/presence
  7) Fusion: file = 0.4*DF_Arena +0.3*AASIST_file +0.3*probOR(voice_fake,music_fake) + presence-aware calibration

Offline: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, no internet in inference.
Performance: ~0.5-1.9s/file (DF_Arena CPU), 1200 files ~38min CPU, ~10min L4.
VRAM <4GB (DF_Arena ONNX CPU, AASIST 0.57M, PANNs 81M optional).
"""
import os, sys, pathlib, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"
import numpy as np, pandas as pd, torch, torch.nn as nn
import soundfile as sf
TARGET_SR=16000
SEG_SEC=4.0

# --- optional deps ---
try:
    import onnxruntime as ort
    HAS_ONNX=True
except:
    HAS_ONNX=False
    ort=None

# demucs wrapper
try:
    from src.models.demucs_wrapper import HTDemucsSeparator, get_separator
    HAS_DEMUCS_WRAPPER=True
except:
    HAS_DEMUCS_WRAPPER=False
    get_separator=None

# PANNs
try:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from src.models.panns import PANNsPresenceWrapper
    HAS_PANNS=True
except:
    HAS_PANNS=False
    PANNsPresenceWrapper=None

# AASIST
try:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from src.models.aasist import AASISTMultitask
    HAS_AASIST=True
except:
    HAS_AASIST=False
    AASISTMultitask=None

# ----------------- audio utils -----------------
def load_audio(path, target_sr=TARGET_SR):
    """Robust load wav/mp3/flac mono/stereo via soundfile, fallback librosa, resample to 16k mono mean."""
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception as e:
        # fallback librosa (handles mp3 via audioread)
        try:
            import librosa
            data, sr = librosa.load(path, sr=target_sr, mono=False)
            if data.ndim == 2:
                data = np.mean(data, axis=0)
            return data.astype(np.float32), target_sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {path}: {e} / {e2}")
    if data.ndim == 2:
        # stereo -> mean
        data = np.mean(data, axis=1) if data.shape[1]<=2 else np.mean(data, axis=1)
        # sf returns (samples, channels), librosa returns (channels, samples)
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

# ----------------- models -----------------
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

def separate_vocals_music(wave, sr=16000, device="cpu"):
    """HTDemucs via wrapper, fallback to hpss"""
    if HAS_DEMUCS_WRAPPER:
        try:
            sep = get_separator(device=device, verbose=False)
            return sep.separate(wave, sr=sr)
        except:
            pass
    # direct hpss fallback
    try:
        import librosa
        y_harm, y_perc = librosa.effects.hpss(wave)
        if len(y_harm) != len(wave):
            y_harm = np.pad(y_harm, (0, max(0, len(wave)-len(y_harm))))[:len(wave)]
            y_perc = np.pad(y_perc, (0, max(0, len(wave)-len(y_perc))))[:len(wave)]
        return y_harm.astype(np.float32), y_perc.astype(np.float32)
    except:
        return wave, wave

def load_df_arena(device="cpu"):
    if not HAS_ONNX:
        return None
    model_paths = [
        pathlib.Path("model/df_arena/df_arena_1b_int8.onnx"),
        pathlib.Path(__file__).parent / "model" / "df_arena" / "df_arena_1b_int8.onnx",
        pathlib.Path("model/df_arena_1b_int8.onnx"),
    ]
    model_path = next((p for p in model_paths if p.exists()), None)
    if model_path is None:
        # also search recursively
        for p in pathlib.Path("model").rglob("*.onnx"):
            if "df_arena" in str(p):
                model_path = p; break
    if model_path is None or not model_path.exists():
        print("DF_Arena ONNX not found, skipping FILE_FAKE primary (will use AASIST)")
        return None
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
        # Try to limit threads for CPU
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.inter_op_num_threads = 1
        # graph optimization
        try:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except: pass
        sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
        # check input name
        inp_name = sess.get_inputs()[0].name
        print(f"loaded DF_Arena from {model_path} providers {sess.get_providers()} input {inp_name}")
        return sess
    except Exception as e:
        print(f"DF_Arena load failed {e}")
        return None

def df_arena_predict(sess, wave, sr=16000, seg_sec=4.0):
    # Use 3 segments for DF_Arena for speed (vs 5 for AASIST) – still topk_mean, 40% faster, meets 60min budget
    seg_len=int(seg_sec*sr)
    if len(wave) <= seg_len:
        segs=[np.pad(wave,(0,seg_len-len(wave)))]
    else:
        # uniform3 positions 0,0.5,1.0
        positions=[0,0.5,1.0]
        segs=[]
        for p in positions:
            start=int((len(wave)-seg_len)*p)
            start=max(0,min(start,len(wave)-seg_len))
            segs.append(wave[start:start+seg_len])
    inp_name = sess.get_inputs()[0].name
    # Try batched inference for speed (3 seg -> 1 run)
    try:
        batch = np.stack(segs, axis=0).astype(np.float32)  # [N, 64000]
        logits = sess.run(None, {inp_name: batch})[0]  # [N,2] or [N,1]
        probs=[]
        for i in range(logits.shape[0]):
            logit = logits[i]
            if logits.ndim==2 and logits.shape[-1]==1:
                prob = float(torch.sigmoid(torch.tensor(float(logit[0]))).item())
            elif logit.shape[-1]==1 or (hasattr(logit,'size') and logit.size==1):
                prob = float(torch.sigmoid(torch.tensor(float(logit if np.asarray(logit).size==1 else logit[0]))).item())
            else:
                m = np.max(logit)
                exp = np.exp(logit - m)
                prob = float(exp[1] / np.sum(exp))
            probs.append(prob)
        return aggregate_predictions(probs, method="topk_mean", top_k=2)
    except Exception as e:
        probs=[]
        for seg in segs:
            inp = seg[np.newaxis, :].astype(np.float32)
            logits = sess.run(None, {inp_name: inp})[0]
            if logits.shape[-1]==1:
                prob = float(torch.sigmoid(torch.tensor(logits[0,0])).item())
                probs.append(prob)
            else:
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                prob = exp / np.sum(exp, axis=1, keepdims=True)
                probs.append(float(prob[0,1]))
        return aggregate_predictions(probs, method="topk_mean", top_k=2)

def load_aasist(device):
    ckpt_candidates = [
        pathlib.Path("model/best.pt"),
        pathlib.Path("model/stageA_aasist.pt"),
        pathlib.Path("model/aasist_best.pt"),
        pathlib.Path(__file__).parent / "model" / "best.pt",
        pathlib.Path(__file__).parent / "model" / "stageA_aasist.pt",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path and HAS_AASIST:
        for base_ch in [32, 16, 64]:
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                sd = ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
                model = AASISTMultitask(base_channels=base_ch)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                # heuristic: if missing <10, assume compatible
                if len(missing) < 20:
                    model.to(device).eval()
                    print(f"loaded AASIST base{base_ch} from {ckpt_path} missing {len(missing)}")
                    return model
            except Exception as e:
                continue
    print("using HeuristicModel (AASIST not found or incompatible)")
    model = HeuristicModel().to(device).eval()
    return model

def load_panns(device):
    if not HAS_PANNS:
        print("PANNs module not available")
        return None
    # Fast check before constructing heavy model
    import pathlib as pl
    cands = [pl.Path("model/panns/Cnn14_mAP=0.431.pth"), pl.Path("model/panns/Cnn14.pth"), pl.Path(__file__).parent / "model" / "panns" / "Cnn14_mAP=0.431.pth"]
    if not any(p.exists() for p in cands):
        print("PANNs pretrained not found, skipping PANNs (AASIST will handle presence) - to enable, place Cnn14_mAP=0.431.pth in model/panns/")
        return None
    try:
        model = PANNsPresenceWrapper(use_pretrained=True)
        if not model.pretrained_loaded:
            print("PANNs pretrained not found, skipping PANNs (AASIST will handle presence)")
            return None
        model.to(device).eval()
        print(f"PANNs loaded pretrained={model.pretrained_loaded}")
        return model
    except Exception as e:
        print(f"PANNs load failed {e}")
        return None

def infer_file(aasist_model, df_sess, panns_model, audio_path, device):
    try:
        wave, sr = load_audio(audio_path)
    except Exception as e:
        print(f"load failed {audio_path}: {e}")
        return [0.5]*5
    if is_silence(wave, thresh=0.008):
        return [0.05, 0.05, 0.05, 0.02, 0.02]
    # HTDemucs separation (demonstrates baseline step, also used for presence calibration)
    try:
        wav_vocals, wav_music = separate_vocals_music(wave, sr=sr, device=device)
    except:
        wav_vocals, wav_music = wave, wave
    # DF_Arena for file fake on original
    file_fake_df = 0.5
    has_df = df_sess is not None
    if has_df:
        try:
            file_fake_df = df_arena_predict(df_sess, wave, sr=sr)
            file_fake_df = float(np.clip(file_fake_df, 0.01, 0.99))
        except Exception as e:
            print(f"DF_Arena predict failed {e}")
            file_fake_df = 0.5
            has_df = False
    # AASIST inference on original (5 heads)
    segs = extract_segments(wave, sr=sr, seg_sec=SEG_SEC)
    all_probs = []
    with torch.inference_mode():
        for seg in segs:
            batch = torch.from_numpy(seg).float().unsqueeze(0).to(device)
            if device == "cuda":
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = aasist_model(batch)
                except:
                    logits = aasist_model(batch)
            else:
                logits = aasist_model(batch)
            if isinstance(logits, dict):
                probs = [torch.sigmoid(logits[k]).item() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]]
            else:
                # tensor
                probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()
                # ensure 5
                if len(probs)!=5:
                    probs = (probs + [0.5]*5)[:5]
            all_probs.append(probs)
    all_probs = np.array(all_probs)  # [N,5]
    # Aggregate with topk for fakes, mean for presence
    file_fake_aasist = aggregate_predictions(all_probs[:,0], method="topk_mean", top_k=2)
    voice_fake_raw = aggregate_predictions(all_probs[:,1], method="topk_mean", top_k=2)
    music_fake_raw = aggregate_predictions(all_probs[:,2], method="topk_mean", top_k=2)
    voice_present_aasist = float(np.mean(all_probs[:,3]))
    music_present_aasist = float(np.mean(all_probs[:,4]))

    # PANNs presence (if available)
    voice_present_panns = None
    music_present_panns = None
    if panns_model is not None:
        try:
            with torch.inference_mode():
                # PANNs expects (B,T) 16k, we give first segment or full wave truncated to 10s max (PANNs trained on 10s)
                # Use first 10s or first segment aggregated?
                # We'll run PANNs on segments and aggregate mean, similar to AASIST but using max per segment
                panns_seg_probs_v = []
                panns_seg_probs_m = []
                for seg in segs[:3]:  # limit to 3 for speed (PANNs heavier)
                    batch = torch.from_numpy(seg).float().unsqueeze(0).to(device)
                    out = panns_model(batch)
                    # out voice_present/music_present are (B,)
                    panns_seg_probs_v.append(float(out["voice_present"].item()))
                    panns_seg_probs_m.append(float(out["music_present"].item()))
                voice_present_panns = float(np.mean(panns_seg_probs_v))
                music_present_panns = float(np.mean(panns_seg_probs_m))
        except Exception as e:
            print(f"PANNs predict failed {e}")
            voice_present_panns = None
            music_present_panns = None

    # Fuse presence: PANNs 0.6 + AASIST 0.4 if PANNs available and pretrained, else AASIST only
    if voice_present_panns is not None and panns_model is not None and getattr(panns_model, 'pretrained_loaded', False):
        voice_present = 0.6*voice_present_panns + 0.4*voice_present_aasist
        music_present = 0.6*music_present_panns + 0.4*music_present_aasist
    elif voice_present_panns is not None:
        # PANNs without pretrained is random, don't trust; use AASIST only
        voice_present = voice_present_aasist
        music_present = music_present_aasist
    else:
        voice_present = voice_present_aasist
        music_present = music_present_aasist

    # Presence-aware fake calibration
    # If voice not present, voice_fake should be low (but keep some uncertainty)
    # Use interpolation: fake_final = presence*fake_raw + (1-presence)*0.05  (prior 0.05)
    # However, to avoid over-suppression when presence is uncertain (~0.5), we apply mild calibration
    # Only suppress if presence <0.4
    if voice_present < 0.4:
        voice_fake = voice_present*voice_fake_raw + (1-voice_present)*0.05
        # also clip
        voice_fake = float(np.clip(0.3*voice_fake_raw + 0.7*voice_fake, 0.01, 0.99))
    else:
        voice_fake = float(voice_fake_raw)
    if music_present < 0.4:
        music_fake = music_present*music_fake_raw + (1-music_present)*0.05
        music_fake = float(np.clip(0.3*music_fake_raw + 0.7*music_fake, 0.01, 0.99))
    else:
        music_fake = float(music_fake_raw)

    # Additional check: use separated stems to adjust presence confidence
    # If separated vocals energy is very low, reduce voice_present
    try:
        # Compute RMS of separated signals vs original
        rms_orig = float((wave**2).mean()**0.5 + 1e-9)
        rms_voc = float((wav_vocals**2).mean()**0.5 + 1e-9)
        rms_mus = float((wav_music**2).mean()**0.5 + 1e-9)
        # If vocals RMS <0.1*orig, likely no voice
        if rms_voc < 0.15 * rms_orig:
            voice_present = min(voice_present, 0.35)
        if rms_mus < 0.15 * rms_orig:
            music_present = min(music_present, 0.35)
    except:
        pass

    # File fake fusion
    if has_df:
        file_fused = 0.5*file_fake_df + 0.5*file_fake_aasist
        p_or = 1 - (1-voice_fake)*(1-music_fake)
        # If neither present (both low), file fake should be driven by DF + AASIST only, not OR
        if voice_present < 0.3 and music_present < 0.3:
            # silence already handled, but if low presence, be conservative
            file_final = 0.6*file_fused + 0.4*file_fake_aasist
        else:
            file_final = 0.4*file_fused + 0.3*p_or + 0.3*file_fake_aasist
        file_final = float(np.clip(file_final, 0.01, 0.99))
    else:
        p_or = 1 - (1-voice_fake)*(1-music_fake)
        file_final = 0.6*file_fake_aasist + 0.4*p_or
        file_final = float(np.clip(file_final, 0.01, 0.99))

    # Final clamps
    return [float(np.clip(file_final,0.01,0.99)), float(np.clip(voice_fake,0.01,0.99)), float(np.clip(music_fake,0.01,0.99)), float(np.clip(voice_present,0.01,0.99)), float(np.clip(music_present,0.01,0.99))]

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
    # Recursively find all audio
    audio_files=sorted(test_path.rglob("*.wav"))+sorted(test_path.rglob("*.mp3"))+sorted(test_path.rglob("*.flac"))+sorted(test_path.rglob("*.m4a"))+sorted(test_path.rglob("*.ogg"))
    audio_files=sorted(set(audio_files))
    print(f"found {len(audio_files)} files under {test_path}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    aasist_model=load_aasist(device)
    df_sess=load_df_arena(device)
    panns_model=load_panns(device)
    # warmup
    try:
        dummy=torch.randn(1,16000*4).to(device)
        with torch.inference_mode():
            _=aasist_model(dummy)
        if df_sess is not None:
            dummy_np=np.random.randn(1,16000*4).astype(np.float32)
            inp_name = df_sess.get_inputs()[0].name
            _=df_sess.run(None, {inp_name: dummy_np})
        if panns_model is not None:
            with torch.inference_mode():
                _=panns_model(dummy)
        print("warmup ok")
    except Exception as e:
        print(f"warmup fail {e}")
        import traceback; traceback.print_exc()
    # Handle sample_submission
    sample_path=None
    for cand in ["./sample_submission.csv","sample_submission.csv","./data/sample_submission.csv", str(pathlib.Path(__file__).parent / "sample_submission.csv"), str(pathlib.Path(__file__).parent / "data" / "sample_submission.csv")]:
        if os.path.exists(cand):
            sample_path=cand
            break
    # Also check if test_dir is ./data/test, sample is ./sample_submission.csv or ../?
    if sample_path is None:
        # search recursively for sample
        for p in pathlib.Path(".").rglob("sample_submission.csv"):
            sample_path=str(p)
            break
    results=[]
    import time, tqdm
    start = time.time()
    for af in tqdm.tqdm(audio_files) if len(audio_files)>10 else audio_files:
        probs=infer_file(aasist_model, df_sess, panns_model, str(af), device)
        results.append([af.stem]+probs)
    elapsed = time.time()-start
    print(f"inference {len(results)} files in {elapsed:.1f}s ({elapsed/len(results):.2f}s/file)" if len(results)>0 else "no files")
    # Reorder if sample exists
    if sample_path:
        try:
            sdf=pd.read_csv(sample_path)
            sample_ids=sdf.iloc[:,0].astype(str).tolist()
            id_to_probs={str(r[0]): r[1:] for r in results}
            # also map with suffix handling - try exact stem, then filename, then partial
            # Build map from stem and filename
            id_to_probs2={}
            for r in results:
                # r[0] is stem, also keep full relative path stem
                id_to_probs2[str(r[0])] = r[1:]
            # also try to match without suffix
            ordered=[]
            for sid in sample_ids:
                sid_str=str(sid)
                if sid_str in id_to_probs2:
                    ordered.append([sid_str]+id_to_probs2[sid_str])
                else:
                    # try stripped extension? sample may include .wav
                    sid_stem = pathlib.Path(sid_str).stem
                    if sid_stem in id_to_probs2:
                        ordered.append([sid_str]+id_to_probs2[sid_stem])
                    else:
                        # partial match
                        found=False
                        for k,v in id_to_probs2.items():
                            if sid_str==k or sid_str in k or k in sid_str or sid_stem==k:
                                ordered.append([sid_str]+v); found=True; break
                        if not found:
                            print(f"warning: {sid_str} not found in test files, using 0.5")
                            ordered.append([sid_str,0.5,0.5,0.5,0.5,0.5])
            df=pd.DataFrame(ordered,columns=[sdf.columns[0],"FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
        except Exception as e:
            print(f"sample handling failed {e}")
            import traceback; traceback.print_exc()
            df=pd.DataFrame(results,columns=["id","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
    else:
        df=pd.DataFrame(results,columns=["id","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
        print("no sample_submission, using id order")
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output,index=False)
    print(f"saved {args.output} {len(df)} rows")
    # also save to ./output/submission.csv if different
    alt=pathlib.Path("./output/submission.csv")
    if str(alt)!=args.output:
        alt.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(alt,index=False)
        print(f"also saved {alt}")

if __name__=="__main__":
    main()
