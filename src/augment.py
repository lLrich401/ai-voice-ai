"""
Domain randomization augmentation for audio deepfake robustness.
Covers: codec, resampling, telephone, filtering, noise, reverb, gain, clipping, etc.
"""
import random
import numpy as np
import io

try:
    import librosa
    HAS_LIBROSA = True
except:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SF = True
except:
    HAS_SF = False

import math

def random_gain(wave, low_db=-12, high_db=6):
    gain_db = random.uniform(low_db, high_db)
    gain = 10**(gain_db/20)
    return np.clip(wave * gain, -1.0, 1.0)

def add_noise(wave, snr_db_range=(5, 30), p=0.5):
    if random.random() > p:
        return wave
    snr_db = random.uniform(*snr_db_range)
    noise = np.random.randn(len(wave)).astype(np.float32)
    sig_power = np.mean(wave**2) + 1e-9
    noise_power = np.mean(noise**2) + 1e-9
    desired_noise_power = sig_power / (10**(snr_db/10))
    scale = np.sqrt(desired_noise_power / noise_power)
    noise = noise * scale
    return np.clip(wave + noise, -1.0, 1.0)

def mp3_compression_aug(wave, sr=16000, bitrates=[32,64,96,128], p=0.5):
    if random.random() > p or not HAS_SF:
        return wave
    try:
        bitrate = random.choice(bitrates)
        buf = io.BytesIO()
        sf.write(buf, wave, sr, format='MP3', subtype='MPEG_LAYER_III')
        buf.seek(0)
        data, _ = sf.read(buf)
        if data.ndim>1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32)
    except Exception as e:
        return low_pass_filter(wave, sr, cutoff=random.uniform(3500, 7000))

def low_pass_filter(wave, sr=16000, cutoff=4000):
    try:
        from scipy.signal import butter, lfilter
        nyq = sr/2
        normal_cutoff = cutoff/nyq
        b,a = butter(4, normal_cutoff, btype='low', analog=False)
        return lfilter(b,a,wave).astype(np.float32)
    except:
        return wave

def high_pass_filter(wave, sr=16000, cutoff=200):
    try:
        from scipy.signal import butter, lfilter
        nyq = sr/2
        normal_cutoff = cutoff/nyq
        b,a = butter(4, normal_cutoff, btype='high', analog=False)
        return lfilter(b,a,wave).astype(np.float32)
    except:
        return wave

def band_pass_filter(wave, sr=16000, low=300, high=3400):
    try:
        from scipy.signal import butter, lfilter
        nyq = sr/2
        low_norm = low/nyq
        high_norm = high/nyq
        b,a = butter(4, [low_norm, high_norm], btype='band')
        return lfilter(b,a,wave).astype(np.float32)
    except:
        return wave

def telephone_simulation(wave, sr=16000, p=0.3):
    if random.random() > p:
        return wave
    wave = band_pass_filter(wave, sr, 300, 3400)
    if HAS_LIBROSA:
        wave_8k = librosa.resample(wave, orig_sr=sr, target_sr=8000)
        wave = librosa.resample(wave_8k, orig_sr=8000, target_sr=sr)
    else:
        from scipy.signal import resample
        n8 = int(len(wave)*8000/sr)
        w8 = resample(wave, n8)
        wave = resample(w8, len(wave))
    wave = mulaw_encode_decode(wave, mu=255, p=0.5 if random.random()<0.5 else 0.0)
    wave = add_noise(wave, snr_db_range=(20,40), p=0.5)
    return wave.astype(np.float32)

def mulaw_encode_decode(wave, mu=255, p=1.0):
    if random.random() > p:
        return wave
    x = np.clip(wave, -1, 1)
    encoded = np.sign(x) * np.log1p(mu*np.abs(x))/np.log1p(mu)
    quantized = np.round((encoded+1)*127.5)/127.5 -1
    quantized = np.clip(quantized, -1, 1)
    decoded = np.sign(quantized) * (1/mu) * ((1+mu)**np.abs(quantized) -1)
    return decoded.astype(np.float32)

def reverb_aug(wave, sr=16000, p=0.2):
    if random.random() > p:
        return wave
    rt60 = random.uniform(0.1, 0.5)
    ir_len = int(rt60 * sr * 0.5)
    ir = np.exp(-3*np.arange(ir_len)/(rt60*sr)) * np.random.randn(ir_len)*0.3
    ir[0]=1
    reverbed = np.convolve(wave, ir, mode='full')[:len(wave)]
    mix = random.uniform(0.1, 0.4)
    out = (1-mix)*wave + mix*reverbed / (np.max(np.abs(reverbed))+1e-6) * np.max(np.abs(wave))
    return np.clip(out, -1, 1).astype(np.float32)

def clipping_aug(wave, p=0.1):
    if random.random() > p:
        return wave
    thresh = random.uniform(0.5, 0.9)
    wave_clipped = np.clip(wave, -thresh, thresh)
    return wave_clipped.astype(np.float32)

def dynamic_range_compression(wave, p=0.2):
    if random.random() > p:
        return wave
    alpha = random.uniform(0.5, 0.9)
    return (np.sign(wave) * (np.abs(wave)**alpha)).astype(np.float32)

class AugmentationPipeline:
    def __init__(self, sr=16000, is_training=True):
        self.sr = sr
        self.is_training = is_training
    def __call__(self, wave):
        if not self.is_training:
            return wave
        wave = random_gain(wave, -6, 6)
        wave = add_noise(wave, (10, 30), p=0.3)
        if random.random() < 0.5:
            wave = band_pass_filter(wave, self.sr, 300, 3400) if random.random()<0.3 else low_pass_filter(wave, self.sr, random.uniform(3000, 6000))
        wave = reverb_aug(wave, self.sr, p=0.15)
        wave = clipping_aug(wave, p=0.05)
        wave = dynamic_range_compression(wave, p=0.1)
        wave = np.clip(wave, -1.0, 1.0)
        return wave.astype(np.float32)
