"""
Audio preprocessing for variable length, codec, channel handling.
Internal: 16kHz mono float32.
Supports: wav, mp3, flac, mono/stereo.
Segment extraction strategies.
"""
import os
import math
import warnings
import numpy as np
import soundfile as sf

TARGET_SR = 16000

def load_audio(path, target_sr=TARGET_SR, mono_mode="mean"):
    try:
        data, sr = sf.read(path, always_2d=False)
    except (RuntimeError, TypeError) as e:
        try:
            import librosa
            data, sr = librosa.load(path, sr=target_sr, mono=False)
            if data.ndim == 2:
                if mono_mode == "left":
                    data = data[0]
                elif mono_mode == "right":
                    data = data[1] if data.shape[0]>1 else data[0]
                elif mono_mode == "mid":
                    data = np.mean(data, axis=0)
                else:
                    data = np.mean(data, axis=0)
            data = data.astype(np.float32)
            if not np.isfinite(data).all():
                raise RuntimeError(f"non-finite samples after librosa fallback: {path}")
            warnings.warn(
                f"soundfile decode failed; used librosa decoder/resampler for {path}",
                RuntimeWarning, stacklevel=2)
            return data, target_sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load {path}: {e} / {e2}")
    if data.ndim == 2:
        n_ch = data.shape[1]
        if n_ch == 1:
            data = data[:,0]
        else:
            if mono_mode == "left":
                data = data[:,0]
            elif mono_mode == "right":
                data = data[:,1]
            elif mono_mode == "mid":
                data = np.mean(data, axis=1)
            elif mono_mode == "side":
                data = (data[:,0] - data[:,1]) / 2.0
            else:
                data = np.mean(data, axis=1)
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            divisor = math.gcd(int(sr), int(target_sr))
            warnings.warn(
                "librosa unavailable; using deterministic scipy.resample_poly fallback",
                RuntimeWarning, stacklevel=2)
            data = resample_poly(data, int(target_sr)//divisor, int(sr)//divisor)
        sr = target_sr
    data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise RuntimeError(f"non-finite samples after loading/resampling: {path}")
    return data, sr

def pad_or_trim(wave, target_len, mode="constant"):
    if len(wave) == target_len:
        return wave
    if len(wave) > target_len:
        start = (len(wave) - target_len)//2
        return wave[start:start+target_len]
    else:
        pad_len = target_len - len(wave)
        if mode == "constant":
            return np.pad(wave, (0, pad_len), mode='constant')
        elif mode == "repeat":
            repeats = math.ceil(target_len/len(wave))
            tiled = np.tile(wave, repeats)[:target_len]
            return tiled
        else:
            return np.pad(wave, (0, pad_len), mode='constant')

def extract_segments(wave, sr=16000, seg_sec=4.0, strategy="uniform5", hop_sec=None):
    seg_len = int(seg_sec * sr)
    total_len = len(wave)
    if total_len <= seg_len:
        return [pad_or_trim(wave, seg_len)]
    segments = []
    if strategy == "center":
        start = (total_len - seg_len)//2
        segments.append(wave[start:start+seg_len])
    elif strategy == "uniform5":
        positions = [0, 0.25, 0.5, 0.75, 1.0]
        for p in positions:
            start = int((total_len - seg_len) * p)
            start = max(0, min(start, total_len - seg_len))
            segments.append(wave[start:start+seg_len])
    elif strategy == "sliding":
        if hop_sec is None:
            hop_sec = seg_sec / 2
        hop = int(hop_sec * sr)
        start = 0
        while start + seg_len <= total_len:
            segments.append(wave[start:start+seg_len])
            start += hop
        if len(segments)==0 or start != total_len - seg_len:
            segments.append(wave[total_len - seg_len: total_len])
    elif strategy == "uniform3":
        positions = [0, 0.5, 1.0]
        for p in positions:
            start = int((total_len - seg_len) * p)
            start = max(0, min(start, total_len - seg_len))
            segments.append(wave[start:start+seg_len])
    elif strategy == "start_end":
        segments.append(wave[0:seg_len])
        segments.append(wave[total_len-seg_len:total_len])
    else:
        return extract_segments(wave, sr, seg_sec, "uniform5", hop_sec)
    return segments

def aggregate_predictions(segment_probs, method="mean", top_k=2):
    probs = np.asarray(segment_probs)
    if probs.ndim == 1:
        probs = probs[:, None]
        squeeze = True
    else:
        squeeze = False
    if method == "mean":
        agg = np.mean(probs, axis=0)
    elif method == "max":
        agg = np.max(probs, axis=0)
    elif method == "topk_mean":
        k = min(top_k, len(probs))
        sorted_probs = np.sort(probs, axis=0)
        topk = sorted_probs[-k:, :]
        agg = np.mean(topk, axis=0)
    elif method == "logit_mean":
        eps = 1e-6
        probs_clipped = np.clip(probs, eps, 1-eps)
        logits = np.log(probs_clipped/(1-probs_clipped))
        mean_logit = np.mean(logits, axis=0)
        agg = 1/(1+np.exp(-mean_logit))
    elif method == "median":
        agg = np.median(probs, axis=0)
    elif method == "trimmed_mean":
        ordered = np.sort(probs, axis=0)
        selected = ordered[1:-1] if len(ordered) > 2 else ordered
        agg = np.mean(selected, axis=0)
    elif method.startswith("max_mean_"):
        alpha = float(method.rsplit("_", 1)[1])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("max/mean alpha must be in [0,1]")
        agg = alpha * np.max(probs, axis=0) + (1.0 - alpha) * np.mean(probs, axis=0)
    elif method == "attention":
        weights = np.abs(probs - 0.5) + 0.1
        weights = weights / np.sum(weights, axis=0, keepdims=True)
        agg = np.sum(probs * weights, axis=0)
    else:
        agg = np.mean(probs, axis=0)
    if squeeze:
        return float(agg[0])
    return agg
