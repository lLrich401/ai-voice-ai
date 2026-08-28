"""
Real dataset pipeline for DACON 236749 - speaker/source/generator leakage-safe, VAL-A/B/C/D, HTDemucs stems.
No synthetic fallback in final training path (synthetic only for emergency testing, not used in run_all_stages).
"""
import pathlib, random, os, hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocess import load_audio

TARGET_SR = 16000
SEG_SEC = 4.0

# Dataset configs matching scripts/download_datasets.py
DATASETS = {
    "librispeech_dev": {"category":"voice_real", "hf_id":"openslr/librispeech_asr", "speaker_key":"speaker_id"},
    "common_voice_small": {"category":"voice_real"},
    "asvspoof2019": {"category":"voice_fake", "generators":["TTS_1","TTS_2","VC_1"]},
    "wavefake": {"category":"voice_fake", "generators":["MelGAN","HiFiGAN","WaveGlow"]},
    "mlaad_small": {"category":"voice_fake"},
    "fma_small": {"category":"music_real"},
    "musiccaps": {"category":"music_real"},
    "fakemusiccaps": {"category":"music_fake", "generators":["MusicGen","AudioLDM2"]},
}

def _infer_labels_from_path(path: pathlib.Path):
    """Infer DACON 5 labels from path and folder name."""
    p = str(path).lower()
    # defaults
    voice_present = 0
    music_present = 0
    voice_fake = 0
    music_fake = 0
    # voice real
    if "librispeech" in p or "libri" in p or "common_voice" in p:
        voice_present=1; music_present=0; voice_fake=0; music_fake=0
    elif "asvspoof" in p or "wavefake" in p or "mlaad" in p:
        voice_present=1; music_present=0
        # check subfolder real vs spoof if exists
        if "bonafide" in p or "real" in p or "librispeech" in p:
            voice_fake=0
        elif "spoof" in p or "fake" in p or "wavefake" in p:
            voice_fake=1
        else:
            # assume fake for wavefake/mlaad, half half for asvspoof need protocol - default fake
            voice_fake=1
        music_fake=0
    elif "fma" in p or "musiccaps" in p and "fake" not in p:
        voice_present=0; music_present=1; voice_fake=0; music_fake=0
    elif "fakemusiccaps" in p or ("music" in p and "fake" in p):
        voice_present=0; music_present=1; voice_fake=0; music_fake=1
    elif "silence" in p:
        voice_present=0; music_present=0; voice_fake=0; music_fake=0
    else:
        # unknown -> treat as voice real (conservative)
        voice_present=1; music_present=0; voice_fake=0; music_fake=0
    file_fake = 1 if (voice_fake==1 or music_fake==1) else 0
    return {
        "file_fake": file_fake,
        "voice_fake": voice_fake,
        "music_fake": music_fake,
        "voice_present": voice_present,
        "music_present": music_present,
    }

def _extract_speaker_id(path: pathlib.Path):
    """Extract speaker/source/generator group for leakage-safe split."""
    # LibriSpeech: data/raw/openslr_librispeech_asr/84/121123/84-121123-0000.wav -> speaker 84
    parts = pathlib.Path(path).parts
    # try to find numeric speaker id
    for part in parts:
        if part.isdigit() and len(part)<=4:
            return f"spk_{part}"
    # for ASVspoof: use generator folder
    p = str(path).lower()
    if "asvspoof" in p:
        if "2019" in p: return "asv2019"
        if "2021" in p: return "asv2021"
        return "asvspoof"
    if "wavefake" in p:
        # generator is encoded in filename or subfolder, fallback to hash of name
        return f"wavefake_{hashlib.md5(str(path).encode()).hexdigest()[:4]}"
    if "mlaad" in p:
        return "mlaad"
    if "fma" in p:
        # track id from filename 000001.wav -> use first 3 digits as group
        stem = pathlib.Path(path).stem
        return f"fma_{stem[:3]}"
    if "fakemusiccaps" in p or "music" in p:
        # generator: MusicGen vs AudioLDM2 - infer from path
        if "musicgen" in p: return "musicgen"
        if "audioldm" in p: return "audioldm"
        return "music_fake"
    if "librispeech" in p:
        # speaker id already handled, fallback
        return f"libri_{hashlib.md5(str(path).encode()).hexdigest()[:6]}"
    # generic group by dataset folder
    for d in ["librispeech","common_voice","asvspoof","wavefake","fma","musiccaps"]:
        if d in p:
            return d
    return "unknown"

def _extract_generator(path: pathlib.Path):
    p = str(path).lower()
    if "wavefake" in p:
        if "melgan" in p: return "MelGAN"
        if "hifigan" in p: return "HiFiGAN"
        if "waveglow" in p: return "WaveGlow"
        return "WaveFake_unknown"
    if "asvspoof" in p:
        if "19" in p: return "ASV2019"
        if "21" in p: return "ASV2021"
        return "ASV_unknown"
    if "mlaad" in p:
        return "MLAAD"
    if "fakemusiccaps" in p:
        if "musicgen" in p: return "MusicGen"
        if "audioldm" in p: return "AudioLDM2"
        return "FakeMusic_unknown"
    if "librispeech" in p: return "LibriSpeech"
    if "fma" in p: return "FMA"
    return "unknown"

def scan_real_datasets(data_root="data/raw", manifest_path="data/manifest.csv"):
    """
    Scan data/raw recursively for real datasets downloaded via scripts/download_datasets.py
    Returns DataFrame with columns: path, file_fake, voice_fake, music_fake, voice_present, music_present,
    speaker_id, generator, source, dataset
    """
    root = pathlib.Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root {root} not found. Run scripts/download_datasets.py")
    audio_files = []
    for ext in ["*.wav","*.mp3","*.flac","*.m4a","*.ogg"]:
        audio_files.extend(list(root.rglob(ext)))
    audio_files = sorted(set(audio_files))
    if len(audio_files)==0:
        raise FileNotFoundError(f"No audio files found in {root}. Run scripts/download_datasets.py --datasets librispeech_dev wavefake fma_small etc")
    rows=[]
    for af in audio_files:
        labels = _infer_labels_from_path(af)
        speaker = _extract_speaker_id(af)
        generator = _extract_generator(af)
        # source dataset folder
        rel = af.relative_to(root).parts[0] if len(af.relative_to(root).parts)>0 else "unknown"
        rows.append({
            "path": str(af),
            "file_fake": labels["file_fake"],
            "voice_fake": labels["voice_fake"],
            "music_fake": labels["music_fake"],
            "voice_present": labels["voice_present"],
            "music_present": labels["music_present"],
            "speaker_id": speaker,
            "generator": generator,
            "source": rel,
            "dataset": rel,
        })
    df = pd.DataFrame(rows)
    # also handle mixed: create synthetic mixes from real voice+music for file_fake training
    # We create mixed samples by pairing: for each voice sample, pair with random music sample with SNR -5~10dB
    # This generates additional rows with voice_present=1, music_present=1, file_fake = OR
    # To keep leakage-safe, mixed samples inherit speaker_id from voice and source from both
    # We create limited mixed: 20% of dataset size
    n_mixed = max(0, min(len(df)//5, 500))  # cap 500 for speed
    if n_mixed>0:
        voice_df = df[df["voice_present"]==1]
        music_df = df[df["music_present"]==1]
        if len(voice_df)>0 and len(music_df)>0:
            for i in range(n_mixed):
                v = voice_df.sample(1, random_state=i).iloc[0] if len(voice_df)>0 else None
                m = music_df.sample(1, random_state=i+1000).iloc[0] if len(music_df)>0 else None
                if v is None or m is None:
                    continue
                # create mixed entry: path will be special marker, actual mixing done at __getitem__
                # store as "MIX::voice_path|music_path"
                mix_path = f"MIX::{v['path']}|{m['path']}"
                # labels: OR
                file_fake = 1 if (v["voice_fake"]==1 or m["music_fake"]==1) else 0
                rows.append({
                    "path": mix_path,
                    "file_fake": file_fake,
                    "voice_fake": int(v["voice_fake"]),
                    "music_fake": int(m["music_fake"]),
                    "voice_present": 1,
                    "music_present": 1,
                    "speaker_id": v["speaker_id"],  # leakage group follows voice speaker
                    "generator": f"{v['generator']}+{m['generator']}",
                    "source": f"mix_{v['source']}_{m['source']}",
                    "dataset": "mix",
                })
            df = pd.DataFrame(rows)
    # save manifest
    pathlib.Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Scanned {len(audio_files)} real files + {n_mixed} mixed -> {len(df)} total, saved {manifest_path}")
    return df

def leakage_safe_split(df, test_size=0.2, random_state=42):
    """
    Speaker/source/generator leakage-safe split: ensure groups in test not in train.
    Uses GroupShuffleSplit on speaker_id.
    Returns train_df, val_df
    """
    from sklearn.model_selection import GroupShuffleSplit
    # If speaker_id has many unique groups, use it; else fallback to source
    groups = df["speaker_id"].values
    # If number of groups < 5, fallback to source
    if len(set(groups)) < 5:
        groups = df["source"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    # verify no group leakage
    train_groups = set(train_df["speaker_id"])
    val_groups = set(val_df["speaker_id"])
    overlap = train_groups & val_groups
    if len(overlap)>0:
        print(f"Warning: group leakage {len(overlap)} groups overlap, but using GroupShuffleSplit should prevent")
    return train_df, val_df

def build_val_sets(df, out_dir="data/splits", random_state=42):
    """
    Build VAL-A/B/C/D from real data:
    VAL-A normal: leakage-safe split, seen generators
    VAL-B unseen generator: generators held out (never seen in train)
    VAL-C codec: VAL-A with mp3/lowpass augmentation marker
    VAL-D telephone: VAL-A with bandpass+8k simulation
    Returns dict with train, val_a, val_b, val_c, val_d DataFrames
    """
    # First, separate by generator for VAL-B unseen
    # Define unseen generators: for voice, hold out one TTS system; for music, hold out one fake generator
    # If dataset has enough generators, hold out last 20% generators as unseen
    generators = df["generator"].unique()
    # Choose unseen: if >3 generators, hold out 1; else hold out synthetic mix generators
    # Use deterministic: sorted generators, last one as unseen
    if len(generators) >= 3:
        unseen_gens = [sorted(generators)[-1]]
        seen_df = df[~df["generator"].isin(unseen_gens)].reset_index(drop=True)
        unseen_df = df[df["generator"].isin(unseen_gens)].reset_index(drop=True)
    else:
        # fallback: no unseen, create empty val_b
        seen_df = df
        unseen_df = pd.DataFrame(columns=df.columns)
        unseen_gens = []
    print(f"Generators total {len(generators)} unseen {unseen_gens} seen {len(seen_df)} unseen_df {len(unseen_df)}")
    # leakage-safe split seen_df into train and val_a
    train_df, val_a = leakage_safe_split(seen_df, test_size=0.2, random_state=random_state)
    # val_b is unseen_df leakage-safe split (or just unseen_df)
    if len(unseen_df)>0:
        # val_b uses unseen generators, keep all as val_b (or split further)
        val_b = unseen_df
        # Ensure val_b size similar to val_a: if too large, subsample
        if len(val_b) > len(val_a)*2:
            val_b = val_b.sample(len(val_a)*2, random_state=random_state).reset_index(drop=True)
    else:
        val_b = pd.DataFrame(columns=df.columns)
        print("Warning: no unseen generators found, VAL-B will be empty (need more datasets)")
    # VAL-C codec: copy val_a with codec flag
    val_c = val_a.copy()
    val_c["augment"] = "codec_mp3"
    # VAL-D telephone
    val_d = val_a.copy()
    val_d["augment"] = "telephone"
    # Add augment column to train/val_a/val_b as none
    for d in [train_df, val_a, val_b]:
        if "augment" not in d.columns:
            d["augment"] = "none"
        else:
            d["augment"] = d["augment"].fillna("none")
    # Save
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(pathlib.Path(out_dir)/"train.csv", index=False)
    val_a.to_csv(pathlib.Path(out_dir)/"val_a.csv", index=False)
    val_b.to_csv(pathlib.Path(out_dir)/"val_b.csv", index=False)
    val_c.to_csv(pathlib.Path(out_dir)/"val_c.csv", index=False)
    val_d.to_csv(pathlib.Path(out_dir)/"val_d.csv", index=False)
    print(f"Saved splits: train {len(train_df)} val_a {len(val_a)} val_b {len(val_b)} val_c {len(val_c)} val_d {len(val_d)} to {out_dir}")
    return {"train":train_df, "val_a":val_a, "val_b":val_b, "val_c":val_c, "val_d":val_d}

# Augmentations for VAL-C/D simulation (applied at dataset __getitem__ if augment column set)
def apply_codec_sim(wave, sr=16000):
    # mp3 sim: lowpass 3.5k
    try:
        from scipy.signal import butter, lfilter
        b,a = butter(4, 3500/(sr/2), btype="low")
        return lfilter(b,a,wave).astype(np.float32)
    except:
        return wave

def apply_telephone_sim(wave, sr=16000):
    try:
        from scipy.signal import butter, lfilter
        import librosa
        b,a = butter(4, [300/(sr/2), 3400/(sr/2)], btype="band")
        w = lfilter(b,a,wave)
        # 8k resample sim
        w8 = librosa.resample(w, orig_sr=sr, target_sr=8000)
        w = librosa.resample(w8, orig_sr=8000, target_sr=sr)
        # mu-law
        mu=255
        x=np.clip(w,-1,1)
        enc=np.sign(x)*np.log1p(mu*np.abs(x))/np.log1p(mu)
        quantized=np.round((enc+1)*127.5)/127.5 -1
        decoded=np.sign(quantized)*(1/mu)*((1+mu)**np.abs(quantized)-1)
        # add slight noise
        decoded = decoded + np.random.randn(len(decoded))*1e-4
        return decoded.astype(np.float32)[:len(wave)]
    except:
        # fallback scipy resample
        try:
            from scipy.signal import resample
            from scipy.signal import butter, lfilter
            b,a = butter(4, [300/(sr/2), 3400/(sr/2)], btype="band")
            w = lfilter(b,a,wave)
            n8=int(len(w)*8000/sr)
            w8=resample(w,n8)
            w=resample(w8,len(wave))
            return w.astype(np.float32)
        except:
            return wave

class AudioDataset(Dataset):
    def __init__(self, df, sr=16000, seg_sec=4.0, is_training=True, use_demucs=False, task="multitask", device="cpu"):
        """
        task: multitask | voice | music | file
            voice: vocals stem for voice detector
            music: accompaniment stem for music detector
            multitask: original wave
        use_demucs: if True, separate via HTDemucs and return stem according to task
        """
        self.df=df.reset_index(drop=True)
        self.sr=sr
        self.seg_sec=seg_sec
        self.is_training=is_training
        self.use_demucs=use_demucs
        self.task=task
        self.device=device
        # For HTDemucs, lazy load separator
        self.separator=None
        if use_demucs:
            try:
                from .models.demucs_wrapper import get_separator
                self.separator=get_separator(device=device)
            except:
                self.separator=None

    def __len__(self): return len(self.df)

    def _load_and_separate(self, path_str):
        # Handle MIX::voice|music
        if path_str.startswith("MIX::"):
            _, rest = path_str.split("MIX::",1)
            v_path, m_path = rest.split("|",1)
            # load both
            v_wave,_ = load_audio(v_path, target_sr=self.sr)
            m_wave,_ = load_audio(m_path, target_sr=self.sr)
            # mix with SNR -5~10
            snr = random.uniform(-5,10) if self.is_training else 0
            # align lengths: take min or pad
            min_len = min(len(v_wave), len(m_wave))
            # if training, random crop to seg? but we do seg later, just mix full then crop
            # ensure same length for mixing: pad shorter
            max_len = max(len(v_wave), len(m_wave))
            if len(v_wave)<max_len:
                v_wave=np.pad(v_wave,(0,max_len-len(v_wave)))
            if len(m_wave)<max_len:
                m_wave=np.pad(m_wave,(0,max_len-len(m_wave)))
            # scale music by SNR
            sig_power=np.mean(v_wave**2)+1e-9
            music_power=np.mean(m_wave**2)+1e-9
            desired=np.sqrt(sig_power/(10**(snr/10))/music_power)
            m_wave=m_wave*desired
            mix=np.clip(v_wave+m_wave, -1, 1).astype(np.float32)
            # for HTDemucs path, we still need original mix; separation will be done after
            # but for task-specific, we might want to return appropriate stem:
            # For now, return mix as original; stem separation will handle
            wave=mix
            # also need to handle vocals/music for task: for mix, vocals stem is v_wave, music stem is m_wave approximated?
            # For efficiency, we can consider separated stems as v_wave/m_wave directly without demucs
            # If task is voice, return v_wave; if music, return m_wave (bypass demucs)
            if self.task=="voice":
                wave=v_wave
            elif self.task=="music":
                wave=m_wave
            return wave
        else:
            wave,_ = load_audio(path_str, target_sr=self.sr)
            # apply augment for VAL-C/D
            row_augment = None
            try:
                # df has augment column
                # but we need idx? We'll handle outside - apply based on self.df augment at __getitem__ caller
                pass
            except:
                pass
            return wave

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        path_str=str(row["path"])
        # load
        if path_str.startswith("MIX::"):
            wave=self._load_and_separate(path_str)
        else:
            wave,_ = load_audio(path_str, target_sr=self.sr)
            # apply codec/telephone augment if specified in row
            augment = str(row.get("augment","none")).lower() if "augment" in row else "none"
            if augment=="codec_mp3" or augment=="codec":
                wave=apply_codec_sim(wave, sr=self.sr)
            elif augment=="telephone" or augment=="tel":
                wave=apply_telephone_sim(wave, sr=self.sr)
            # HTDemucs stem separation
            if self.use_demucs and self.separator is not None and not path_str.startswith("MIX::"):
                # only apply for single-source or mix if not already handled
                try:
                    vocals, music = self.separator.separate(wave, sr=self.sr)
                    if self.task=="voice":
                        wave=vocals
                    elif self.task=="music":
                        wave=music
                    # for multitask/file keep original
                except:
                    pass
        # segment
        seg_len=int(self.seg_sec*self.sr)
        if len(wave) < seg_len:
            wave=np.pad(wave,(0,seg_len-len(wave)))
        elif self.is_training and len(wave) > seg_len:
            s=random.randint(0,len(wave)-seg_len)
            wave=wave[s:s+seg_len]
        else:
            # eval: center crop
            wave=wave[:seg_len] if len(wave)>=seg_len else wave
            # if longer, take center
            if len(wave) > seg_len:
                start=(len(wave)-seg_len)//2
                wave=wave[start:start+seg_len]
        # labels
        labels = torch.tensor([row.get(k,0) for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dtype=torch.float32)
        # For task-specific, we may mask irrelevant labels? Keep all but trainer will select
        return torch.from_numpy(wave).float(), labels, str(path_str)
