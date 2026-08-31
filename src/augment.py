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


def _fast_fir(wave, impulse):
    """Length-preserving FIR convolution without quadratic NumPy cost."""
    try:
        from scipy.signal import fftconvolve
        return fftconvolve(wave, impulse, mode="full")[:len(wave)]
    except Exception:
        return np.convolve(wave, impulse, mode="full")[:len(wave)]

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

def alaw_encode_decode(wave, a=87.6, p=1.0):
    """Approximate 8-bit G.711 A-law companding without external codecs."""
    if random.random() > p:
        return wave
    x = np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0)
    magnitude = np.abs(x)
    denominator = 1.0 + np.log(a)
    compressed = np.where(
        magnitude < (1.0 / a),
        a * magnitude / denominator,
        (1.0 + np.log(a * np.maximum(magnitude, 1e-12))) / denominator,
    )
    encoded = np.sign(x) * compressed
    quantized = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
    qmag = np.abs(quantized)
    expanded = np.where(
        qmag < (1.0 / denominator),
        qmag * denominator / a,
        np.exp(qmag * denominator - 1.0) / a,
    )
    return (np.sign(quantized) * expanded).astype(np.float32)


def signed_linear_quantize(wave, bits=8):
    """Linear PCM stress quantizer that preserves signed zero exactly."""
    maximum = float((1 << (int(bits) - 1)) - 1)
    if maximum < 1:
        raise ValueError("quantization requires at least 2 bits")
    source = np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0)
    return np.clip(np.round(source * maximum) / maximum, -1.0, 1.0).astype(np.float32)

def resample_roundtrip(wave, sr=16000, target_rates=(8000, 11025, 12000), p=0.25):
    """Simulate unknown capture rates while returning the canonical sample rate."""
    if random.random() > p:
        return wave
    target = int(random.choice(target_rates))
    try:
        from scipy.signal import resample_poly
        import math as _math
        down_gcd = _math.gcd(sr, target)
        low = resample_poly(wave, target // down_gcd, sr // down_gcd)
        up_gcd = _math.gcd(target, sr)
        restored = resample_poly(low, sr // up_gcd, target // up_gcd)
        if len(restored) < len(wave):
            restored = np.pad(restored, (0, len(wave) - len(restored)))
        return np.asarray(restored[:len(wave)], dtype=np.float32)
    except Exception:
        return wave

def short_dropout(wave, sr=16000, max_duration_sec=0.04, p=0.1):
    """Packet-loss-like short attenuation; intentionally moderate."""
    if random.random() > p or len(wave) == 0:
        return wave
    maximum = max(1, min(len(wave), int(round(max_duration_sec * sr))))
    length = random.randint(1, maximum)
    start = random.randint(0, max(0, len(wave) - length))
    result = np.asarray(wave, dtype=np.float32).copy()
    result[start:start + length] *= random.uniform(0.0, 0.15)
    return result

def reverb_aug(wave, sr=16000, p=0.2):
    if random.random() > p:
        return wave
    rt60 = random.uniform(0.1, 0.5)
    ir_len = int(rt60 * sr * 0.5)
    ir = np.exp(-3*np.arange(ir_len)/(rt60*sr)) * np.random.randn(ir_len)*0.3
    ir[0]=1
    reverbed = _fast_fir(wave, ir)
    mix = random.uniform(0.1, 0.4)
    out = (1-mix)*wave + mix*reverbed / (np.max(np.abs(reverbed))+1e-6) * np.max(np.abs(wave))
    return np.clip(out, -1, 1).astype(np.float32)


def rerecording_simulation(wave, sr=16000, p=0.25):
    """Moderate loudspeaker-room-microphone simulation.

    This is deliberately a compact channel model rather than a synthetic fake
    generator.  It preserves speech content and synthesis traces while adding
    a short room response, speaker/microphone coloration, mild non-linearity,
    and a realistic low noise floor.
    """
    if random.random() > p or len(wave) == 0:
        return wave
    source = np.asarray(wave, dtype=np.float32)
    ir_length = max(8, int(round(random.uniform(0.025, 0.12) * sr)))
    impulse = np.zeros(ir_length, dtype=np.float32)
    impulse[0] = 1.0
    for _ in range(random.randint(2, 6)):
        delay = random.randint(max(1, int(0.003 * sr)), ir_length - 1)
        impulse[delay] += random.uniform(-0.35, 0.35) * np.exp(-3.0 * delay / ir_length)
    tail = np.random.randn(ir_length).astype(np.float32)
    tail *= np.exp(-5.0 * np.arange(ir_length, dtype=np.float32) / ir_length)
    impulse += tail * random.uniform(0.002, 0.015)
    recorded = _fast_fir(source, impulse)
    recorded = high_pass_filter(recorded, sr, random.uniform(60.0, 180.0))
    recorded = low_pass_filter(recorded, sr, random.uniform(4200.0, 7600.0))
    drive = random.uniform(1.0, 1.8)
    recorded = np.tanh(recorded * drive) / np.tanh(drive)
    signal_rms = float(np.sqrt(np.mean(recorded ** 2) + 1e-12))
    noise_rms = signal_rms / (10.0 ** (random.uniform(28.0, 45.0) / 20.0))
    recorded += np.random.randn(len(recorded)).astype(np.float32) * noise_rms
    return np.clip(recorded, -1.0, 1.0).astype(np.float32)

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
    def __init__(self, sr=16000, is_training=True, profile="baseline"):
        self.sr = sr
        self.is_training = is_training
        self.profile = str(profile)
    def __call__(self, wave):
        if not self.is_training:
            return wave
        if self.profile == "voice_channel_v10":
            # One primary channel is chosen for every source/class. Avoiding a
            # stack of unrelated heavy transforms preserves synthesis traces
            # and prevents the augmentation recipe itself becoming a shortcut.
            channels = (
                "clean", "clean", "clean",
                "codec_variable", "codec_narrow",
                "telephone_mulaw", "telephone_alaw", "telephone_narrow",
                "telephone_lowbit", "rerecord",
            )
            channel = random.choice(channels)
            if channel == "codec_variable":
                wave = low_pass_filter(wave, self.sr, random.uniform(4200.0, 7000.0))
                if random.random() < 0.6:
                    wave = resample_roundtrip(
                        wave, self.sr, target_rates=(11025, 12000), p=1.0)
                wave = signed_linear_quantize(wave, random.choice((10, 12)))
            elif channel == "codec_narrow":
                wave = low_pass_filter(wave, self.sr, random.uniform(3000.0, 3600.0))
                wave = resample_roundtrip(wave, self.sr, target_rates=(8000,), p=1.0)
                wave = signed_linear_quantize(wave, 8)
            elif channel.startswith("telephone_"):
                low, high = ((400.0, 3000.0) if channel == "telephone_narrow"
                             else (300.0, 3400.0))
                wave = band_pass_filter(wave, self.sr, low, high)
                wave = resample_roundtrip(wave, self.sr, target_rates=(8000,), p=1.0)
                if channel == "telephone_mulaw" or channel == "telephone_narrow":
                    wave = mulaw_encode_decode(wave, p=1.0)
                elif channel == "telephone_alaw":
                    wave = alaw_encode_decode(wave, p=1.0)
                else:
                    # Deliberately low-probability severe robustness stress;
                    # this is not represented as a bit-exact GSM codec.
                    wave = signed_linear_quantize(wave, 6)
            elif channel == "rerecord":
                wave = rerecording_simulation(wave, self.sr, p=1.0)
            if random.random() < 0.7:
                wave = random_gain(wave, -4.0, 4.0)
            wave = add_noise(wave, (18, 40), p=0.2)
            wave = reverb_aug(wave, self.sr, p=0.08)
            wave = short_dropout(wave, self.sr, max_duration_sec=0.04, p=0.05)
            return np.clip(wave, -1.0, 1.0).astype(np.float32)
        wave = random_gain(wave, -6, 6)
        wave = add_noise(wave, (10, 30), p=0.3)
        if random.random() < 0.5:
            wave = band_pass_filter(wave, self.sr, 300, 3400) if random.random()<0.3 else low_pass_filter(wave, self.sr, random.uniform(3000, 6000))
        wave = reverb_aug(wave, self.sr, p=0.15)
        wave = clipping_aug(wave, p=0.05)
        wave = dynamic_range_compression(wave, p=0.1)
        if self.profile in ("voice_channel_v7", "voice_channel_v9"):
            # Exactly one primary channel family is selected so the synthesis
            # trace is not destroyed by a stack of unrealistically strong
            # transformations. Remaining perturbations are deliberately mild.
            channels = ["clean", "telephone", "mulaw", "alaw", "low_bitrate"]
            if self.profile == "voice_channel_v9":
                # Two slots give re-recording meaningful exposure without
                # stacking it on every sample or erasing generator artifacts.
                channels.extend(("rerecord", "rerecord"))
            channel = random.choice(channels)
            if channel == "telephone":
                wave = telephone_simulation(wave, self.sr, p=1.0)
            elif channel == "mulaw":
                wave = mulaw_encode_decode(wave, p=1.0)
            elif channel == "alaw":
                wave = alaw_encode_decode(wave, p=1.0)
            elif channel == "low_bitrate":
                wave = low_pass_filter(wave, self.sr, random.uniform(2800, 5200))
            elif channel == "rerecord":
                wave = rerecording_simulation(wave, self.sr, p=1.0)
            wave = resample_roundtrip(
                wave, self.sr,
                target_rates=(8000, 11025, 12000, 22050, 44100),
                p=0.45 if self.profile == "voice_channel_v9" else 0.35,
            )
            wave = short_dropout(
                wave, self.sr, max_duration_sec=0.06,
                p=0.12 if self.profile == "voice_channel_v9" else 0.08,
            )
        elif self.profile != "baseline":
            raise ValueError(f"Unknown augmentation profile: {self.profile}")
        wave = np.clip(wave, -1.0, 1.0)
        return wave.astype(np.float32)
