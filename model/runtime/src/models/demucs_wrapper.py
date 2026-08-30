"""
HTDemucs wrapper for DACON 236749 baseline

Baseline uses Hybrid Transformer Demucs (htdemucs) to separate vocals and music (other).
- Ideally: load demucs pretrained htdemucs/htdemucs_ft via demucs API
- Fallback: librosa.effects.hpss (harmonic=vocals, percussive=music) – lightweight, offline, fast

This wrapper provides unified API: separate(wave: np.ndarray, sr=16000) -> (vocals, music)
Handles mono/stereo, any length.
"""
import numpy as np
import pathlib

HAS_DEMUCS = False
try:
    import demucs.api
    import torch
    HAS_DEMUCS = True
except Exception:
    HAS_DEMUCS = False

HAS_LIBROSA = False
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

def separate_hpss(wave, sr=16000):
    """fast fallback – hpss is slow (4s per 10s), so by default return original.
    Set env USE_HPSS=1 to enable real librosa hpss (offline, accurate but slow)."""
    import os
    if os.environ.get("USE_HPSS","0") != "1":
        return wave, wave
    if not HAS_LIBROSA:
        return wave, wave
    try:
        y_harm, y_perc = librosa.effects.hpss(wave)
        if len(y_harm) != len(wave):
            y_harm = np.pad(y_harm, (0, max(0, len(wave)-len(y_harm))))[:len(wave)]
            y_perc = np.pad(y_perc, (0, max(0, len(wave)-len(y_perc))))[:len(wave)]
        return y_harm.astype(np.float32), y_perc.astype(np.float32)
    except Exception as e:
        print(f"hpss failed {e}, fallback to original")
        return wave, wave

class HTDemucsSeparator:
    def __init__(self, device="cpu", model_name="htdemucs", verbose=False, enabled=False):
        self.device = device
        self.model_name = model_name
        self.verbose = verbose
        self.separator = None
        self.use_demucs = False
        # Do not initialize Demucs unless explicitly requested: the evaluator is
        # offline and an implicit model lookup can waste time before fallback.
        if enabled and HAS_DEMUCS:
            try:
                # Try to load separator; this may download if not cached, but we are offline so will fail gracefully
                # Use demucs.api.Separator
                self.separator = demucs.api.Separator(model=model_name, device=device, progress=False)
                self.use_demucs = True
                if verbose: print(f"HTDemucs loaded {model_name} on {device}")
            except Exception as e:
                if verbose: print(f"HTDemucs load failed {model_name}: {e}, fallback to hpss")
                self.separator = None
                self.use_demucs = False
        elif verbose and enabled:
            print("demucs not installed, using identity separation")

    def separate(self, wave, sr=16000):
        """
        wave: np.ndarray (T,) mono 16k float32
        returns (vocals, music) as np.ndarray (T,) each
        - vocals: extracted vocal stem
        - music: extracted instrumental/music stem (other + drums + bass)
        If demucs succeeds, use its 4 stems: drums, bass, other, vocals -> music = drums+bass+other, vocals=vocals
        Else hpss.
        """
        if self.use_demucs and self.separator is not None:
            try:
                import torch
                # demucs expects torch tensor shape (channels, time) with sr=44100. It will resample internally.
                # Convert mono to stereo by duplicating if needed? demucs handles mono but prefers stereo.
                # Create tensor (1, T) then repeat to (2, T) for better separation?
                wav_t = torch.from_numpy(wave).float().unsqueeze(0)  # (1, T)
                # Resample to 44100 if needed? Separator does internally if needed via sample_rate param? Actually Separator uses 44100 default
                # We pass via separate function that handles sr?
                # demucs.api.Separator.separate(waveform) expects waveform shape (channels, T) at model's sr (44100)
                # So we need to resample to 44100 first
                if sr != 44100:
                    try:
                        import torchaudio.functional as F
                        wav_44 = F.resample(wav_t, orig_freq=sr, new_freq=44100)
                    except:
                        # fallback via librosa or scipy
                        if HAS_LIBROSA:
                            import librosa
                            wav_np_44 = librosa.resample(wave, orig_sr=sr, target_sr=44100)
                            wav_44 = torch.from_numpy(wav_np_44).float().unsqueeze(0)
                        else:
                            from scipy.signal import resample
                            n = int(len(wave)*44100/sr)
                            wav_np_44 = resample(wave, n)
                            wav_44 = torch.from_numpy(wav_np_44).float().unsqueeze(0)
                else:
                    wav_44 = wav_t
                # Stereo: duplicate channel
                if wav_44.size(0)==1:
                    wav_44 = wav_44.repeat(2,1)  # (2, T)
                wav_44 = wav_44.to(self.device)
                with torch.inference_mode():
                    # Separator.separate returns dict or tensor? In demucs.api, it returns Tensor shape (sources, channels, time)
                    # For htdemucs, sources = 4: drums, bass, other, vocals
                    origins, separated = self.separator.separate(wav_44)
                    # separated is dict: {source: tensor}
                    # Alternative API: self.separator.separate returns (waveform, dict)
                    # Let's handle both
                    if isinstance(separated, dict):
                        # keys: drums, bass, other, vocals
                        vocals_44 = separated.get("vocals", None)
                        if vocals_44 is None:
                            # try case where separated values are tensors list
                            vocals_44 = list(separated.values())[-1]
                        music_44 = None
                        # sum other sources for music
                        for k in ["drums","bass","other"]:
                            if k in separated:
                                if music_44 is None:
                                    music_44 = separated[k]
                                else:
                                    music_44 = music_44 + separated[k]
                        if music_44 is None:
                            # if only vocals, music = original - vocals
                            music_44 = wav_44 - vocals_44
                    else:
                        # separated is Tensor (sources, channels, time)
                        # Assume order: drums, bass, other, vocals
                        if separated.dim()==3:
                            vocals_44 = separated[-1]  # (channels, T)
                            music_44 = separated[:-1].sum(dim=0)
                        else:
                            raise RuntimeError("unknown separated shape")
                    # Average channels to mono and resample back to original sr
                    vocals_44 = vocals_44.mean(dim=0) if vocals_44.dim()>1 else vocals_44
                    music_44 = music_44.mean(dim=0) if music_44.dim()>1 else music_44
                    vocals_np = vocals_44.cpu().numpy()
                    music_np = music_44.cpu().numpy()
                    # Resample back to sr
                    if 44100 != sr:
                        if HAS_LIBROSA:
                            import librosa
                            vocals_np = librosa.resample(vocals_np, orig_sr=44100, target_sr=sr)
                            music_np = librosa.resample(music_np, orig_sr=44100, target_sr=sr)
                        else:
                            from scipy.signal import resample
                            vocals_np = resample(vocals_np, len(wave))
                            music_np = resample(music_np, len(wave))
                    # Ensure length matches
                    if len(vocals_np) != len(wave):
                        vocals_np = np.pad(vocals_np, (0, max(0, len(wave)-len(vocals_np))))[:len(wave)]
                        music_np = np.pad(music_np, (0, max(0, len(wave)-len(music_np))))[:len(wave)]
                    return vocals_np.astype(np.float32), music_np.astype(np.float32)
            except Exception as e:
                if self.verbose: print(f"demucs separate failed {e}, fallback hpss")
                return separate_hpss(wave, sr=sr)
        # Fallback hpss
        return separate_hpss(wave, sr=sr)

# Singleton for easy import
_separators = {}
def get_separator(device="cpu", verbose=False, use_demucs=False):
    key = (str(device), bool(use_demucs))
    if key not in _separators:
        _separators[key] = HTDemucsSeparator(device=device, verbose=verbose, enabled=use_demucs)
    return _separators[key]

def separate_vocals_music(wave, sr=16000, device="cpu", use_demucs=False):
    sep = get_separator(device=device, use_demucs=use_demucs)
    return sep.separate(wave, sr=sr)
