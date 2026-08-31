#!/usr/bin/env python3
"""
DACON 236749 Real-Data Pipeline Inference: PANNs + HTDemucs + DF_Arena_1B + Voice/Music Detectors
- Voice detector (AASIST) on vocals stem via HTDemucs
- Music detector (SpecCNN) on music stem via HTDemucs
- PANNs presence, DF_Arena file fake, fusion weights optimized on VAL
- Fails clearly if mandatory models missing (no 0.5 fallback)
- Exact ID mapping only
"""
import os, sys, pathlib, warnings, json, hashlib
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"

# The official archive permits only model/, script.py and requirements.txt at
# its top level. Runtime source is bundled inside model/runtime/.
_RUNTIME_ROOT = pathlib.Path(__file__).resolve().parent / "model" / "runtime"
if _RUNTIME_ROOT.exists():
    sys.path.insert(0, str(_RUNTIME_ROOT))

import numpy as np, pandas as pd, torch
import soundfile as sf
TARGET_SR=16000
SEG_SEC=4.0
OUTPUT_EPS=1e-6
PIPELINE_VERSION="dacon236749-20260831-v5"
# DF-Arena model card: input length 64,600 at 16 kHz and logits ordered as
# [spoof, bonafide].  FAKE therefore always maps to class index 0.
DF_INPUT_SAMPLES=64600
DF_SEG_SEC=DF_INPUT_SAMPLES/TARGET_SR
DF_ARENA_LABELS=("spoof", "bonafide")
DF_ARENA_FAKE_INDEX=0

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
        ("DF_Arena_ORT", pathlib.Path("model/df_arena/df_arena_1b_int8.ort")),
        ("DF_Arena", pathlib.Path("model/df_arena/df_arena_1b_int8.onnx")),
        ("DF_Arena_alt", pathlib.Path(__file__).parent / "model" / "df_arena" / "df_arena_1b_int8.onnx"),
        ("PANNs", pathlib.Path("model/panns/Cnn14_mAP=0.431.pth")),
        ("PANNs_alt", pathlib.Path(__file__).parent / "model" / "panns" / "Cnn14_mAP=0.431.pth"),
        ("Voice_checkpoint", pathlib.Path("model/best.pt")),
        ("Music_checkpoint", pathlib.Path("model/music_best.pt")),
        ("Fusion_weights", pathlib.Path("model/fusion_weights.json")),
    ]
    # Check existence with fallback
    df_exists = checks[0][1].exists() or checks[1][1].exists() or checks[2][1].exists()
    panns_exists = checks[3][1].exists() or checks[4][1].exists()
    voice_exists = checks[5][1].exists()
    music_exists = checks[6][1].exists()
    fusion_exists = checks[7][1].exists()
    if not df_exists:
        missing.append("model/df_arena/df_arena_1b_int8.onnx (1.37GB DF_Arena_1B)")
    if not panns_exists:
        missing.append("model/panns/Cnn14_mAP=0.431.pth (PANNs CNN14)")
    if not voice_exists:
        missing.append("model/best.pt (voice detector checkpoint)")
    if not music_exists:
        missing.append("model/music_best.pt (music detector checkpoint)")
    if not fusion_exists:
        missing.append("model/fusion_weights.json (validated fusion weights)")
    if missing:
        raise FileNotFoundError(f"Mandatory calibrated artifacts missing: {missing}")
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

def select_aux_segments(wave, sr=16000, seg_sec=4.0):
    """Adaptive 1/2/3-crop policy for short/medium/long recordings."""
    candidates=extract_segments(wave,sr=sr,seg_sec=seg_sec)
    energies=[float(np.mean(np.asarray(seg,dtype=np.float32)**2)) for seg in candidates]
    duration=len(wave)/float(sr)
    count=1 if duration<=8.0 else (2 if duration<=25.0 else 3)
    selected=sorted(np.argsort(energies)[-min(count,len(candidates)):].tolist())
    return [candidates[i] for i in selected]


def limit_aux_segments(segments, maximum):
    if maximum is None or int(maximum) <= 0 or len(segments) <= int(maximum):
        return segments
    energies=[float(np.mean(np.asarray(segment,dtype=np.float32)**2)) for segment in segments]
    selected=sorted(np.argsort(energies)[-int(maximum):].tolist())
    return [segments[index] for index in selected]


# Backwards-compatible name for external notebooks.
select_aux_segment=select_aux_segments

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
        pathlib.Path("model/df_arena/df_arena_1b_int8.ort"),
        pathlib.Path(__file__).parent / "model" / "df_arena" / "df_arena_1b_int8.onnx",
    ]
    model_path=next((p for p in paths if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("DF_Arena ORT/ONNX model not found under model/df_arena")
    # Dynamic INT8 MatMul/Gemm is CPU-optimised.  Sending the graph through the
    # CUDA EP can split it between CPU/GPU and add transfer overhead.  The
    # Keep the complete graph on CPU and use the cores actually exposed by the
    # evaluator. This is 6 on the official machine and scales to local hosts.
    providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads=max(1,min(int(os.cpu_count() or 1),16))
    sess_opts.inter_op_num_threads=1
    sess_opts.enable_mem_pattern=True
    sess_opts.enable_cpu_mem_arena=True
    try:
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except: pass
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    print(f"loaded DF_Arena {model_path} {sess.get_providers()} input {sess.get_inputs()[0].name}")
    return sess

def df_arena_fake_probability(logits):
    """Convert DF-Arena logits to FAKE probability using class 0 (spoof)."""
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim==1:
        logits=logits[None,:]
    if logits.shape[-1]!=2:
        raise ValueError(f"DF_Arena expected 2 logits {DF_ARENA_LABELS}, got {logits.shape}")
    shifted=logits-np.max(logits,axis=-1,keepdims=True)
    exp=np.exp(shifted)
    return exp[:,DF_ARENA_FAKE_INDEX]/np.sum(exp,axis=-1)

def df_arena_predict(sess, wave, sr=16000, seg_sec=DF_SEG_SEC):
    segs=_df_arena_segments(wave,sr,seg_sec)
    inp_name = sess.get_inputs()[0].name
    # batched
    batch = np.stack(segs, axis=0).astype(np.float32)
    logits = sess.run(None, {inp_name: batch})[0]
    probs=df_arena_fake_probability(logits)
    return aggregate_predictions(probs, method="topk_mean", top_k=2)

def _df_arena_crop_candidates(wave, sr=16000, seg_sec=DF_SEG_SEC):
    """Return primary/secondary high-energy crops and their start samples."""
    seg_len=int(round(seg_sec*sr))
    if len(wave) <= seg_len:
        # Tile-repeat rather than zero-pad, matching DF-Arena preprocessing.
        if len(wave)==0:
            crop=np.zeros(seg_len,dtype=np.float32)
            return crop,None,0,None
        repeats=(seg_len+len(wave)-1)//len(wave)
        crop=np.tile(wave,repeats)[:seg_len]
        return crop,None,0,None
    # Scan 1-second hops using cumulative energy and choose the strongest
    # window.  This avoids feeding silence while requiring only one 1B-model
    # forward pass per file.
    hop=max(sr,1)
    starts=list(range(0,len(wave)-seg_len+1,hop))
    if starts[-1] != len(wave)-seg_len:
        starts.append(len(wave)-seg_len)
    squared=np.asarray(wave,dtype=np.float64)**2
    cumulative=np.concatenate(([0.0],np.cumsum(squared)))
    energies=[cumulative[start+seg_len]-cumulative[start] for start in starts]
    best=starts[int(np.argmax(energies))]
    minimum_distance=max(seg_len//2,2*sr)
    eligible=[(energy,start) for energy,start in zip(energies,starts)
              if abs(start-best)>=minimum_distance]
    second=max(eligible)[1] if eligible else None
    return wave[best:best+seg_len],(wave[second:second+seg_len] if second is not None else None),best,second


def _df_arena_segments(wave, sr=16000, seg_sec=DF_SEG_SEC):
    primary,_,_,_=_df_arena_crop_candidates(wave,sr,seg_sec)
    return [primary]


def should_use_adaptive_df_second_crop(duration_sec, primary_fake_probability,
                                       low=0.25, high=0.75, minimum_duration=12.0):
    return (float(duration_sec)>=float(minimum_duration)
            and float(low)<float(primary_fake_probability)<float(high))

def df_arena_predict_batch(sess, waves, sr=16000, seg_sec=DF_SEG_SEC,
                           adaptive_config=None, return_details=False):
    """Run DF_Arena for several files in one ONNX Runtime call.

    This preserves the per-file crops and top-k aggregation from
    ``df_arena_predict`` while substantially reducing GPU launch overhead.
    """
    config={"enabled":True,"low":0.25,"high":0.75,"minimum_duration":12.0,
            "aggregation":"mean","force_second_for_long":False}
    if adaptive_config:
        config.update(adaptive_config)
    candidates=[_df_arena_crop_candidates(wave,sr,seg_sec) for wave in waves]
    primary_batch=np.stack([item[0] for item in candidates]).astype(np.float32)
    primary=df_arena_fake_probability(
        sess.run(None,{sess.get_inputs()[0].name:primary_batch})[0])
    selected=[]
    for index,(wave,item,score) in enumerate(zip(waves,candidates,primary)):
        has_second=item[1] is not None
        trigger=config["enabled"] and has_second and (
            config.get("force_second_for_long",False)
            or should_use_adaptive_df_second_crop(len(wave)/sr,score,config["low"],config["high"],config["minimum_duration"]))
        if trigger:
            selected.append(index)
    second_scores={}
    if selected:
        second_batch=np.stack([candidates[index][1] for index in selected]).astype(np.float32)
        second_probs=df_arena_fake_probability(
            sess.run(None,{sess.get_inputs()[0].name:second_batch})[0])
        second_scores=dict(zip(selected,map(float,second_probs)))
    results=[]; details=[]
    for index,score in enumerate(primary):
        second=second_scores.get(index)
        if second is None:
            combined=float(score)
        elif config["aggregation"]=="max":
            combined=max(float(score),second)
        else:
            combined=(float(score)+second)/2.0
        results.append(combined)
        details.append({"primary":float(score),"second":second,
                        "primary_start":candidates[index][2],"second_start":candidates[index][3],
                        "used_second":second is not None})
    return (results,details) if return_details else results

def _strict_checkpoint_model(ckpt_path, device, expected_task):
    ckpt=torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt,dict) or "model" not in ckpt or "backbone" not in ckpt:
        raise RuntimeError(f"{ckpt_path}: checkpoint metadata/model state missing")
    task=str(ckpt.get("task", expected_task))
    if task not in (expected_task, "multitask"):
        raise RuntimeError(f"{ckpt_path}: task={task}, expected {expected_task}")
    required=("model_name","base_channels","sample_rate","seg_sec","label_heads","epoch","selection_score")
    missing=[key for key in required if key not in ckpt]
    if missing:
        raise RuntimeError(f"{ckpt_path}: required checkpoint config missing: {missing}")
    if int(ckpt["sample_rate"])!=TARGET_SR or tuple(ckpt["label_heads"]) != ("file_fake","voice_fake","music_fake","voice_present","music_present"):
        raise RuntimeError(f"{ckpt_path}: incompatible sample rate or label heads")
    backbone=str(ckpt["backbone"])
    base_channels=int(ckpt["base_channels"])
    if backbone=="aasist":
        model=AASISTMultitask(base_channels=base_channels)
    elif backbone=="spec_cnn":
        model=MusicMultitask(base_channels=base_channels)
    else:
        raise RuntimeError(f"{ckpt_path}: unsupported backbone={backbone}")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    print(f"loaded {expected_task} {backbone} strictly from {ckpt_path}")
    return model


def load_voice_model(device):
    cands=[pathlib.Path("model/best.pt"), pathlib.Path("model/voice_aasist.pt"), pathlib.Path(__file__).parent/"model"/"best.pt"]
    ckpt_path=next((p for p in cands if p.exists()), None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Voice checkpoint not found: model/best.pt (train via scripts/run_all_stages.py)")
    return _strict_checkpoint_model(ckpt_path, device, "voice")

def load_music_model(device):
    cands=[pathlib.Path("model/music_best.pt"), pathlib.Path(__file__).parent/"model"/"music_best.pt"]
    ckpt_path=next((p for p in cands if p.exists()), None)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("Music checkpoint not found: model/music_best.pt (train via scripts/run_all_stages.py)")
    return _strict_checkpoint_model(ckpt_path, device, "music")

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
        expected={
            "pipeline_version":PIPELINE_VERSION,
            "voice_checkpoint_sha256":_sha256_file(pathlib.Path("model/best.pt")),
            "music_checkpoint_sha256":_sha256_file(pathlib.Path("model/music_best.pt")),
        }
        stale={key:(w.get(key),value) for key,value in expected.items() if w.get(key)!=value}
        if stale:
            raise RuntimeError(f"Stale fusion weights; recalibration required: {stale}")
        # Keep the detector sub-ensemble on a proper convex combination even
        # when a hand-edited JSON file contains unnormalised values.
        detector_keys=("w_voice_file", "w_music_file", "w_prob_or")
        try:
            detector=np.asarray([float(w.get(k, d)) for k,d in zip(detector_keys, (0.5,0.3,0.2))], dtype=np.float64)
            if not np.isfinite(detector).all() or (detector < 0).any() or detector.sum() <= 0:
                raise ValueError("invalid detector weights")
            detector=detector/detector.sum()
            for key, value in zip(detector_keys, detector):
                w[key]=float(value)
        except (TypeError, ValueError):
            print("Warning: invalid detector fusion weights; using 0.5/0.3/0.2")
            w.update({"w_voice_file":0.5, "w_music_file":0.3, "w_prob_or":0.2})
        # DF_Arena has a separately calibrated output scale.  Its blend is
        # explicit so that a future validation run can tune it without a code
        # change; 0.5 preserves the previously validated submission behaviour.
        try:
            w["w_df_arena"]=float(np.clip(float(w.get("w_df_arena", 0.5)), 0.0, 1.0))
        except (TypeError, ValueError):
            w["w_df_arena"]=0.5
        legacy=w.get("w_df_component",0.0)
        for key in ("w_df_voice_component","w_df_music_component"):
            try:
                w[key]=float(np.clip(float(w.get(key,legacy)),0.0,1.0))
            except (TypeError,ValueError):
                w[key]=0.0
        print(f"loaded fusion {w}")
        return w
    raise FileNotFoundError("model/fusion_weights.json is mandatory; run validation calibration first")


def _sha256_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def _run_torch_segments(model, segment_groups, device, use_amp=True, outputs_are_logits=True):
    """Run one model over all segments from a file batch, retaining bounds."""
    bounds=[]; flat=[]
    for segs in segment_groups:
        start=len(flat); flat.extend(segs); bounds.append((start,len(flat)))
    batch=torch.from_numpy(np.stack(flat)).float()
    if device=="cuda":
        batch=batch.pin_memory().to(device,non_blocking=True)
    else:
        batch=batch.to(device)
    with torch.inference_mode():
        if device=="cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output=model(batch)
        else:
            output=model(batch)
    if outputs_are_logits:
        output={key:torch.sigmoid(value).detach().float().cpu().numpy() for key,value in output.items()}
    else:
        output={key:value.detach().float().cpu().numpy() for key,value in output.items()}
    return output, bounds

def fuse_prediction_features(file_fake_df, voice_fake_model, music_fake_model,
                             file_voice, file_music, voice_present_model,
                             music_present_model, voice_present_panns,
                             music_present_panns, fusion_weights):
    """Canonical score fusion shared by validation and submitted inference."""
    values=np.nan_to_num(np.asarray([
        file_fake_df,voice_fake_model,music_fake_model,file_voice,file_music,
        voice_present_model,music_present_model,voice_present_panns,music_present_panns,
    ],dtype=np.float64),nan=0.5,posinf=1.0,neginf=0.0)
    (file_fake_df,voice_fake_model,music_fake_model,file_voice,file_music,
     voice_present_model,music_present_model,voice_present_panns,music_present_panns)=values
    w_panns=float(fusion_weights.get("w_panns_presence",0.6))
    voice_present=w_panns*voice_present_panns + (1.0-w_panns)*voice_present_model
    music_present=w_panns*music_present_panns + (1.0-w_panns)*music_present_model
    # The generic DF_Arena score is the robust fake signal for this external
    # domain.  The small in-domain heads remain a secondary cue rather than
    # being allowed to overturn it on unseen generators.
    legacy=fusion_weights.get("w_df_component",0.0)
    w_voice_component=fusion_weights.get("w_df_voice_component",legacy)
    w_music_component=fusion_weights.get("w_df_music_component",legacy)
    voice_fake=w_voice_component*file_fake_df+(1.0-w_voice_component)*voice_fake_model
    music_fake=w_music_component*file_fake_df+(1.0-w_music_component)*music_fake_model
    wv=fusion_weights.get("w_voice_file",0.5)
    wm=fusion_weights.get("w_music_file",0.3)
    wo=fusion_weights.get("w_prob_or",0.2)
    prob_or=1-(1-voice_fake)*(1-music_fake)
    file_fusion_mode=str(fusion_weights.get("file_fusion_mode","legacy"))
    if file_fusion_mode in ("presence_weighted","presence_component_or"):
        # Presence changes FILE risk only. The official component EERs are
        # conditional, so VOICE_FAKE_PROB/MUSIC_FAKE_PROB themselves must keep
        # their ungated rankings.
        voice_risk=voice_present*voice_fake
        music_risk=music_present*music_fake
        component_or=1-(1-voice_risk)*(1-music_risk)
        if file_fusion_mode=="presence_component_or":
            detector_fused=component_or
        else:
            voice_file_risk=voice_present*file_voice
            music_file_risk=music_present*file_music
            detector_fused=wv*voice_file_risk + wm*music_file_risk + wo*component_or
    elif file_fusion_mode=="legacy":
        detector_fused=wv*file_voice + wm*file_music + wo*prob_or
    else:
        raise ValueError(f"unsupported file_fusion_mode={file_fusion_mode}")
    w_df=fusion_weights.get("w_df_arena",0.5)
    file_final=float(np.clip(w_df*file_fake_df+(1.0-w_df)*detector_fused,OUTPUT_EPS,1.0-OUTPUT_EPS))
    return [file_final, float(np.clip(voice_fake,OUTPUT_EPS,1.0-OUTPUT_EPS)), float(np.clip(music_fake,OUTPUT_EPS,1.0-OUTPUT_EPS)), float(np.clip(voice_present,OUTPUT_EPS,1.0-OUTPUT_EPS)), float(np.clip(music_present,OUTPUT_EPS,1.0-OUTPUT_EPS))]


def _combine_predictions(file_fake_df, v_probs, m_probs, panns_out, fusion_weights):
    """Aggregate segments, then call the one canonical fusion function."""
    return fuse_prediction_features(
        file_fake_df,
        aggregate_predictions(v_probs[:,1], method="topk_mean", top_k=2),
        aggregate_predictions(m_probs[:,2], method="topk_mean", top_k=2),
        aggregate_predictions(v_probs[:,0], method="topk_mean", top_k=2),
        aggregate_predictions(m_probs[:,0], method="topk_mean", top_k=2),
        float(np.mean(v_probs[:,3])), float(np.mean(m_probs[:,4])),
        float(np.mean(panns_out["voice_present"])),
        float(np.mean(panns_out["music_present"])), fusion_weights)


def infer_wave_features_batch(voice_model, music_model, df_sess, panns_model,
                              waves, device, use_demucs=False, df_config=None,
                              specialist_max_segments=None, panns_max_segments=None):
    """Canonical preprocessing/model/aggregation path used by val and submit."""
    separator=get_separator(device=device, use_demucs=use_demucs)
    separator_type="htdemucs" if getattr(separator,"use_demucs",False) else "identity"
    if use_demucs and separator_type!="htdemucs":
        raise RuntimeError(f"HTDemucs requested but unavailable (separator_type={separator_type})")
    segment_groups_v=[]; segment_groups_m=[]; segment_groups_o=[]
    for wave in waves:
        vocals,music=separator.separate(wave,sr=TARGET_SR)
        segment_groups_v.append(limit_aux_segments(
            select_aux_segments(vocals,sr=TARGET_SR,seg_sec=SEG_SEC),specialist_max_segments))
        segment_groups_m.append(limit_aux_segments(
            select_aux_segments(music,sr=TARGET_SR,seg_sec=SEG_SEC),specialist_max_segments))
        segment_groups_o.append(limit_aux_segments(
            select_aux_segments(wave,sr=TARGET_SR,seg_sec=SEG_SEC),panns_max_segments))
    v_out,v_bounds=_run_torch_segments(voice_model,segment_groups_v,device,use_amp=True)
    m_out,m_bounds=_run_torch_segments(music_model,segment_groups_m,device,use_amp=True)
    p_out,p_bounds=_run_torch_segments(panns_model,segment_groups_o,device,use_amp=False,outputs_are_logits=False)
    gate_threshold=(df_config or {}).get("gate_voice_presence_threshold")
    if gate_threshold is None:
        df_indices=list(range(len(waves)))
    else:
        df_indices=[]
        for row,(start,end) in enumerate(v_bounds):
            if float(np.mean(v_out["voice_present"][start:end])) >= float(gate_threshold):
                df_indices.append(row)
    df_probs=[0.5]*len(waves)
    df_details=[{"primary":0.5,"second":None,"primary_start":None,
                 "second_start":None,"used_second":False} for _ in waves]
    if df_indices:
        selected_probs,selected_details=df_arena_predict_batch(
            df_sess,[waves[index] for index in df_indices],sr=TARGET_SR,seg_sec=DF_SEG_SEC,
            adaptive_config=df_config,return_details=True)
        for index,probability,detail in zip(df_indices,selected_probs,selected_details):
            df_probs[index]=probability
            df_details[index]=detail
    df_used=set(df_indices)
    features=[]
    for row in range(len(waves)):
        va,vb=v_bounds[row]; ma,mb=m_bounds[row]; pa,pb=p_bounds[row]
        features.append({
            "df":float(df_probs[row]),
            "df_primary":float(df_details[row]["primary"]),
            "df_second":float(df_details[row]["second"]) if df_details[row]["second"] is not None else np.nan,
            "df_has_second":bool(df_details[row]["second"] is not None),
            "df_used":row in df_used,
            "duration_sec":float(len(waves[row])/TARGET_SR),
            "vf":aggregate_predictions(v_out["voice_fake"][va:vb],"topk_mean",2),
            "mf":aggregate_predictions(m_out["music_fake"][ma:mb],"topk_mean",2),
            "vfile":aggregate_predictions(v_out["file_fake"][va:vb],"topk_mean",2),
            "mfile":aggregate_predictions(m_out["file_fake"][ma:mb],"topk_mean",2),
            "vp_model":float(np.mean(v_out["voice_present"][va:vb])),
            "mp_model":float(np.mean(m_out["music_present"][ma:mb])),
            "vp_panns":float(np.mean(p_out["voice_present"][pa:pb])),
            "mp_panns":float(np.mean(p_out["music_present"][pa:pb])),
        })
    return features


def fuse_feature_record(feature, fusion_weights):
    row_weights=fusion_weights
    if not feature.get("df_used",True):
        row_weights=dict(fusion_weights)
        row_weights["w_df_voice_component"]=0.0
        row_weights["w_df_music_component"]=0.0
    return fuse_prediction_features(
        feature["df"],feature["vf"],feature["mf"],feature["vfile"],feature["mfile"],
        feature["vp_model"],feature["mp_model"],feature["vp_panns"],feature["mp_panns"],
        row_weights)

def infer_files_batch(voice_model, music_model, df_sess, panns_model, fusion_weights, audio_paths, device, use_demucs=False):
    """Inference for a small group of files with batched GPU model calls."""
    separator=get_separator(device=device, use_demucs=use_demucs)
    separator_type="htdemucs" if getattr(separator,"use_demucs",False) else "identity"
    if not hasattr(infer_files_batch,"_logged_sep"):
        print(f"separator_type={separator_type} use_demucs={use_demucs}; file batch inference enabled")
        infer_files_batch._logged_sep=True
    results=[None]*len(audio_paths); records=[]
    # Audio decoding/resampling is independent and the official machine has
    # six CPU cores.  Load the next inference batch concurrently, then reserve
    # all six cores for DF-Arena's sequential INT8 graph.
    from concurrent.futures import ThreadPoolExecutor
    workers=min(6,len(audio_paths))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded=list(pool.map(lambda path:load_audio(str(path)),audio_paths))
    for index,(audio_path,(wave,sr)) in enumerate(zip(audio_paths,loaded)):
        records.append((index,audio_path,wave))
    if not records:
        return results
    df_config={
        "enabled":bool(fusion_weights.get("adaptive_df_enabled",True)),
        "low":float(fusion_weights.get("adaptive_df_low",0.25)),
        "high":float(fusion_weights.get("adaptive_df_high",0.75)),
        "aggregation":str(fusion_weights.get("adaptive_df_aggregation","mean")),
    }
    if fusion_weights.get("df_gate_policy") == "voice_presence":
        df_config["gate_voice_presence_threshold"]=float(
            fusion_weights.get("df_gate_voice_presence_threshold",0.8))
    features=infer_wave_features_batch(
        voice_model,music_model,df_sess,panns_model,[r[2] for r in records],device,use_demucs,
        df_config=df_config,
        specialist_max_segments=int(fusion_weights.get("specialist_max_segments",3)),
        panns_max_segments=int(fusion_weights.get("panns_max_segments",3)))
    for (index,audio_path,_),feature in zip(records,features):
        results[index]=[audio_path.stem]+fuse_feature_record(feature,fusion_weights)
    return results

def infer_file(voice_model, music_model, df_sess, panns_model, fusion_weights,
               audio_path, device, use_demucs=False):
    """Single-file adapter over the canonical batch implementation."""
    row=infer_files_batch(
        voice_model,music_model,df_sess,panns_model,fusion_weights,
        [pathlib.Path(audio_path)],device,use_demucs=use_demucs)[0]
    return row[1:]


def order_results_by_sample(results, sample_ids):
    """Exact ID/stem mapping in sample order; ambiguous/missing IDs are fatal."""
    id_to_probs={str(r[0]):r[1:] for r in results}
    if len(id_to_probs)!=len(results):
        raise ValueError("Duplicate test file stems make exact ID mapping ambiguous")
    ordered=[]
    for sid in map(str,sample_ids):
        key=sid if sid in id_to_probs else pathlib.Path(sid).stem
        if key not in id_to_probs:
            raise FileNotFoundError(
                f"sample ID {sid} not found in test files {list(id_to_probs)[:5]} (exact mapping)")
        ordered.append([sid]+id_to_probs[key])
    return ordered


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="./data/test")
    parser.add_argument("--output", default="./output/submission.csv")
    parser.add_argument("--use_demucs", action="store_true", help="use HTDemucs separation; if set, must be available else fail (requirement 7)")
    parser.add_argument("--batch_files", type=int, default=16, help="files per inference batch; tuned for the official L4 22.4GB server")
    args=parser.parse_args()
    # Log separator type expectation
    if args.use_demucs:
        print("Requested --use_demucs, will verify HTDemucs is loaded")
    else:
        print("Running without HTDemucs (identity separation, unified train/inference) - requirement 7 alternative")
    # Verify mandatory models first (fail fast)
    verify_mandatory_models()
    requested=pathlib.Path(args.test_dir)
    candidates=[requested, pathlib.Path("./data/test"), pathlib.Path("./open"), pathlib.Path("./data")]
    test_path=next((p for p in candidates if p.exists() and any(
        next(p.rglob(ext),None) is not None for ext in ("*.wav","*.mp3","*.flac","*.m4a","*.ogg"))), None)
    if test_path is None:
        raise FileNotFoundError(f"No official input directory found among {[str(p) for p in candidates]}")
    audio_files=sorted(test_path.rglob("*.wav"))+sorted(test_path.rglob("*.mp3"))+sorted(test_path.rglob("*.flac"))+sorted(test_path.rglob("*.m4a"))+sorted(test_path.rglob("*.ogg"))
    audio_files=sorted(set(audio_files))
    if len(audio_files)==0:
        raise FileNotFoundError(f"No audio files found in {test_path}")
    print(f"found {len(audio_files)} files under {test_path}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    if args.batch_files < 1:
        raise ValueError("--batch_files must be at least 1")
    if device=="cuda":
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
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
    dummy_np=np.random.randn(1,DF_INPUT_SAMPLES).astype(np.float32)
    _=df_sess.run(None, {df_sess.get_inputs()[0].name: dummy_np})
    print("warmup ok")
    # sample_submission exact mapping only
    sample_path=None
    for cand in [test_path.parent/"sample_submission.csv", pathlib.Path("./data/sample_submission.csv"), pathlib.Path("./open/sample_submission.csv"), pathlib.Path("./sample_submission.csv"), pathlib.Path(__file__).parent/"sample_submission.csv"]:
        if pathlib.Path(cand).exists():
            sample_path=cand; break
    # Do not recursively pick up another competition's sample submission from
    # a neighbouring project directory. The official contract uses one of the
    # explicit paths above.
    results=[]
    import time, tqdm
    start=time.time()
    batch_starts=range(0,len(audio_files),args.batch_files)
    iterator=tqdm.tqdm(batch_starts,total=(len(audio_files)+args.batch_files-1)//args.batch_files,unit="batch") if len(audio_files)>10 else batch_starts
    for start_index in iterator:
        results.extend(infer_files_batch(voice_model,music_model,df_sess,panns_model,fusion_weights,audio_files[start_index:start_index+args.batch_files],device,use_demucs=args.use_demucs))
    elapsed=time.time()-start
    print(f"inference {len(results)} files in {elapsed:.1f}s ({elapsed/len(results):.2f}s/file)" if len(results)>0 else "no files")
    # Exact mapping only (no substring)
    if sample_path:
        sdf=pd.read_csv(sample_path)
        sample_ids=sdf.iloc[:,0].astype(str).tolist()
        ordered=order_results_by_sample(results,sample_ids)
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
