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
    Load official manifest if exists (written by scripts/download_datasets.py with HF metadata),
    otherwise scan data/raw and fallback to path inference (warning: not official).
    Official manifest contains HF metadata: file_fake, voice_fake, music_fake, voice_present, music_present,
    speaker_id, generator, source, dataset, hf_id, original_id - created by downloader preserving ASVspoof bonafide/spoof,
    WaveFake generator, FakeMusicCaps real/fake & generator.
    """
    mp = pathlib.Path(manifest_path)
    # If manifest exists and was created by new downloader (has hf_id column), use it directly
    if mp.exists():
        try:
            df_existing = pd.read_csv(mp)
            if "hf_id" in df_existing.columns and len(df_existing)>0:
                # Legacy manifests could contain pre-split mixes.  They are unsafe:
                # the same source recording can then land in train and validation.
                df_existing = df_existing[~df_existing["path"].astype(str).str.startswith("MIX::")].reset_index(drop=True)
                # Verify files still exist (except MIX)
                # Filter to existing files
                mask = df_existing["path"].apply(lambda p: p.startswith("MIX::") or pathlib.Path(p).exists())
                if mask.all():
                    print(f"Loaded official manifest {mp} {len(df_existing)} rows (HF metadata preserved, no path inference)")
                    # Ensure dtypes
                    for c in ["file_fake","voice_fake","music_fake","voice_present","music_present"]:
                        if c in df_existing.columns:
                            df_existing[c] = df_existing[c].astype(int)
                    return df_existing
                else:
                    missing = (~mask).sum()
                    print(f"Manifest {mp} has {missing} missing files, rescanning and merging")
                    # Keep existing rows that exist, rescan missing?
                    df_existing = df_existing[mask].reset_index(drop=True)
                    # Continue to scan for additional files not in manifest
                    existing_paths = set(df_existing["path"].tolist())
                # Fall through to scan additional files, but keep official labels for existing
            else:
                print(f"Manifest {mp} exists but no hf_id column (legacy path-inference), will rescan with official logic")
                existing_paths = set()
                df_existing = pd.DataFrame()
        except Exception as e:
            print(f"Failed to load manifest {mp}: {e}, rescanning")
            existing_paths = set()
            df_existing = pd.DataFrame()
    else:
        existing_paths = set()
        df_existing = pd.DataFrame()

    # Scan files for any new files not in manifest
    root = pathlib.Path(data_root)
    if not root.exists():
        if len(df_existing)>0:
            print(f"data_root {root} not found, returning existing manifest {len(df_existing)} rows")
            return df_existing
        raise FileNotFoundError(f"data_root {root} not found. Run scripts/download_datasets.py")
    audio_files = []
    for ext in ["*.wav","*.mp3","*.flac","*.m4a","*.ogg"]:
        audio_files.extend(list(root.rglob(ext)))
    audio_files = sorted(set(audio_files))
    if len(audio_files)==0 and len(df_existing)==0:
        raise FileNotFoundError(f"No audio files found in {root}. Run scripts/download_datasets.py --datasets librispeech_dev wavefake fma_small etc")
    # Filter to files not already in manifest
    new_files = [af for af in audio_files if str(af) not in existing_paths]
    if len(new_files)==0 and len(df_existing)>0:
        print(f"No new files, using existing manifest {len(df_existing)} rows")
        return df_existing
    print(f"Scanning {len(new_files)} new files not in manifest (fallback path inference - WARNING: should use downloader for official labels)")
    rows=[]
    for af in new_files:
        labels = _infer_labels_from_path(af)
        speaker = _extract_speaker_id(af)
        generator = _extract_generator(af)
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
            "hf_id": "path_inference_fallback",
            "original_id": af.stem,
        })
    df_new = pd.DataFrame(rows)
    if len(df_existing)>0 and len(df_new)>0:
        df = pd.concat([df_existing, df_new], ignore_index=True)
    elif len(df_existing)>0:
        df = df_existing
    else:
        df = df_new
    # save manifest
    pathlib.Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Scanned {len(audio_files)} real files -> {len(df)} originals, saved {manifest_path}")
    return df


MIX_MODES = ("simultaneous", "voice_then_music", "music_then_voice", "partial_overlap", "crossfade")
MIX_SNRS_DB = (-10.0, -5.0, 0.0, 5.0, 10.0)


def mixed_labels(voice_fake, music_fake):
    """Return official [file, voice, music, voice_present, music_present] labels."""
    vf, mf = int(voice_fake), int(music_fake)
    return [int(vf or mf), vf, mf, 1, 1]


def _row_source_ids(row):
    """All original recordings represented by an original or generated mix row."""
    ids = set()
    for key in ("base_voice_id", "base_music_id"):
        value = str(row.get(key, ""))
        if value and value.lower() != "nan":
            ids.add(value)
    if not ids:
        value = str(row.get("original_id", ""))
        if value and value.lower() != "nan":
            ids.add(value)
    return ids


def assert_no_base_source_overlap(left, right, names=("left", "right")):
    left_ids = set().union(*(_row_source_ids(r) for _, r in left.iterrows())) if len(left) else set()
    right_ids = set().union(*(_row_source_ids(r) for _, r in right.iterrows())) if len(right) else set()
    overlap = left_ids & right_ids
    assert not overlap, f"{names[0]}/{names[1]} base-source overlap: {len(overlap)} {sorted(overlap)[:3]}"


def add_split_internal_mixes(split_df, mixes_per_class=40, random_state=42):
    """Add balanced RR/RF/FR/FF mixes using only recordings already in one split."""
    base = split_df[~split_df["path"].astype(str).str.startswith("MIX::")].copy().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    pools = {}
    for component, present_col, fake_col in (
        ("voice", "voice_present", "voice_fake"),
        ("music", "music_present", "music_fake"),
    ):
        for fake in (0, 1):
            pool = base[(base[present_col] == 1) & (base[fake_col] == fake)]
            if pool.empty:
                raise ValueError(f"Cannot create balanced mixes: {component} fake={fake} pool is empty")
            pools[(component, fake)] = pool.reset_index(drop=True)

    rows = []
    for vf in (0, 1):
        for mf in (0, 1):
            vp, mp = pools[("voice", vf)], pools[("music", mf)]
            for i in range(int(mixes_per_class)):
                v = vp.iloc[int(rng.integers(len(vp)))]
                m = mp.iloc[int(rng.integers(len(mp)))]
                mode = MIX_MODES[(i + 2 * vf + mf) % len(MIX_MODES)]
                snr = MIX_SNRS_DB[(i + vf + 2 * mf) % len(MIX_SNRS_DB)]
                labels = mixed_labels(vf, mf)
                v_id, m_id = str(v["original_id"]), str(m["original_id"])
                rows.append({
                    "path": f"MIX::{v['path']}|{m['path']}",
                    "file_fake": labels[0], "voice_fake": labels[1], "music_fake": labels[2],
                    "voice_present": 1, "music_present": 1,
                    "speaker_id": f"mix::{v['speaker_id']}::{m['speaker_id']}",
                    "generator": f"mix::{v['generator']}::{m['generator']}",
                    "source": "split_internal_mix", "dataset": "mix", "hf_id": "generated_after_split",
                    "original_id": f"mix::{v_id}::{m_id}::{vf}{mf}::{i}",
                    "base_voice_id": v_id, "base_music_id": m_id,
                    "mix_mode": mode, "mix_snr_db": snr, "mix_crossfade_sec": 0.25,
                })
    originals = base.copy()
    originals["base_voice_id"] = originals.apply(
        lambda r: str(r["original_id"]) if int(r["voice_present"]) else "", axis=1)
    originals["base_music_id"] = originals.apply(
        lambda r: str(r["original_id"]) if int(r["music_present"]) else "", axis=1)
    return pd.concat([originals, pd.DataFrame(rows)], ignore_index=True, sort=False)


def render_mixed_wave(voice_wave, music_wave, mode="simultaneous", snr_db=0.0,
                      crossfade_sec=0.25, sr=TARGET_SR):
    """Render simultaneous/sequential/overlap mixtures deterministically."""
    voice = np.asarray(voice_wave, dtype=np.float32)
    music = np.asarray(music_wave, dtype=np.float32)
    v_power = float(np.mean(voice ** 2)) + 1e-9
    m_power = float(np.mean(music ** 2)) + 1e-9
    music = music * np.sqrt(v_power / (10.0 ** (float(snr_db) / 10.0) * m_power))

    def overlap_at(first, second, offset):
        out = np.zeros(max(len(first), offset + len(second)), dtype=np.float32)
        out[:len(first)] += first
        out[offset:offset + len(second)] += second
        return out

    if mode == "simultaneous":
        out = overlap_at(voice, music, 0)
    elif mode == "voice_then_music":
        out = np.concatenate([voice, music])
    elif mode == "music_then_voice":
        out = np.concatenate([music, voice])
    elif mode == "partial_overlap":
        out = overlap_at(voice, music, max(1, len(voice) // 2))
    elif mode == "crossfade":
        fade = min(max(1, int(round(crossfade_sec * sr))), len(voice), len(music))
        out = np.concatenate([voice[:-fade], voice[-fade:] * np.linspace(1, 0, fade, dtype=np.float32)
                              + music[:fade] * np.linspace(0, 1, fade, dtype=np.float32), music[fade:]])
    else:
        raise ValueError(f"Unknown mix mode: {mode}")
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)
    return out.astype(np.float32)

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

def _get_utterance_group(row):
    """Extract original utterance group for leakage check: for WaveFake use LJ ID, for others use original_id."""
    oid = str(row.get("original_id",""))
    # WaveFake: original_id like wavefake_ajay_LJ002-0032 -> LJ002-0032
    if "LJ" in oid:
        # extract LJxxx-xxxx
        import re
        m = re.search(r"LJ\d+-\d+", oid)
        if m:
            return m.group(0)
    # For GTZAN etc, use original_id as is
    return oid

def build_val_sets(df, out_dir="data/splits", random_state=42):
    """
    Build VAL-A/B/C/D from real data with explicit holdout and leakage-safe guarantees:
    VAL-A normal: leakage-safe split on speaker_id, seen generators
    VAL-B unseen generator: explicit voice fake (WF7) + music fake (AudioLDM2) held-out, original_id utterance separated, positive/negative asserted
    VAL-C codec: VAL-A with mp3/lowpass
    VAL-D telephone: VAL-A with bandpass+8k
    """
    # Explicit holdout: voice fake generator and music fake generator
    voice_fake_gens = sorted(df[df["voice_fake"]==1]["generator"].unique().tolist())
    music_fake_gens = sorted(df[df["music_fake"]==1]["generator"].unique().tolist())
    print(f"voice_fake generators: {voice_fake_gens}")
    print(f"music_fake generators: {music_fake_gens}")
    holdout_voice = None
    holdout_music = None
    # Prefer WF7 for voice, AudioLDM2 for music if present
    if "WF7" in voice_fake_gens:
        holdout_voice = "WF7"
    elif len(voice_fake_gens)>0:
        holdout_voice = voice_fake_gens[-1]
    if "AudioLDM2" in music_fake_gens:
        holdout_music = "AudioLDM2"
    elif "MusicGen" in music_fake_gens:
        # hold out MusicGen if AudioLDM2 not present
        holdout_music = "MusicGen"
    elif len(music_fake_gens)>0:
        holdout_music = music_fake_gens[-1]
    unseen_gens = []
    if holdout_voice:
        unseen_gens.append(holdout_voice)
    if holdout_music:
        unseen_gens.append(holdout_music)
    # If still empty (no fake music), fallback to at least one generator
    if len(unseen_gens)==0:
        generators = sorted(df["generator"].unique())
        unseen_gens = [generators[-1]]
        print(f"No voice/music fake found, fallback holdout {unseen_gens}")
    print(f"Explicit holdout generators voice={holdout_voice} music={holdout_music} => unseen {unseen_gens}")

    # Separate unseen_df (fake holdout) and seen_df
    unseen_df = df[df["generator"].isin(unseen_gens)].copy().reset_index(drop=True)
    seen_df = df[~df["generator"].isin(unseen_gens)].copy().reset_index(drop=True)
    print(f"Generators total {len(df['generator'].unique())} unseen {unseen_gens} seen {len(seen_df)} unseen_df {len(unseen_df)}")

    # Original_id / utterance separation: ensure no original utterance in train appears in VAL-B with different generator
    # For WaveFake, same LJ utterance may have multiple WF versions; all versions of that LJ must be together
    # Collect unseen utterance groups
    unseen_utterances = set(unseen_df.apply(_get_utterance_group, axis=1).tolist()) if len(unseen_df)>0 else set()
    # Also collect original_id directly for strict check
    unseen_original_ids = set(unseen_df["original_id"].tolist()) if len(unseen_df)>0 else set()
    print(f"Unseen utterances {len(unseen_utterances)} example {list(unseen_utterances)[:3]}")
    # Filter seen_df to exclude any row whose utterance group is in unseen
    if len(unseen_utterances)>0:
        mask = seen_df.apply(lambda r: _get_utterance_group(r) in unseen_utterances, axis=1)
        # Also check original_id exact
        mask2 = seen_df["original_id"].isin(unseen_original_ids)
        mask = mask | mask2
        removed = mask.sum()
        if removed>0:
            print(f"Removing {removed} rows from seen_df to prevent original_id leakage to VAL-B (same LJ utterance)")
            seen_df = seen_df[~mask].reset_index(drop=True)

    # Leakage-safe split seen_df into train and val_a on speaker_id
    train_df, val_a = leakage_safe_split(seen_df, test_size=0.2, random_state=random_state)

    # Additional check: ensure train and val_a have no speaker_id overlap (already via leakage_safe_split)
    # Build VAL-B: unseen fake + real negatives to ensure positive/negative both present
    if len(unseen_df)>0:
        # Need to add real samples to VAL-B to have both classes
        # Real negatives: sample from df where file_fake==0 and not in train/val_a utterance groups
        train_utterances = set(train_df.apply(_get_utterance_group, axis=1).tolist()) if len(train_df)>0 else set()
        train_original_ids = set(train_df["original_id"].tolist())
        val_a_utterances = set(val_a.apply(_get_utterance_group, axis=1).tolist()) if len(val_a)>0 else set()
        # Pool for real negatives: all real not in train (val_a overlap allowed for real, but train/val_b must be 0)
        real_pool = df[(df["file_fake"]==0) & (~df["original_id"].isin(train_original_ids))]
        # Also exclude unseen utterances already
        real_pool = real_pool[~real_pool.apply(lambda r: _get_utterance_group(r) in unseen_utterances, axis=1)]
        # Sample real to balance: aim for 1:1 ratio with fake, at least 20% real
        n_fake = len(unseen_df)
        n_real_needed = max(min(n_fake, len(real_pool)), 20)  # at least 20 real
        # If music fake holdout, need both voice and music real? Sample diverse
        # Take 50% voice real, 50% music real if available
        voice_real_pool = real_pool[real_pool["voice_present"]==1]
        music_real_pool = real_pool[real_pool["music_present"]==1]
        # Sample
        import random as _rnd
        _rnd.seed(random_state)
        val_b_real = []
        if len(voice_real_pool)>0 and len(music_real_pool)>0:
            n_voice = min(len(voice_real_pool), n_real_needed//2 + n_real_needed%2)
            n_music = min(len(music_real_pool), n_real_needed//2)
            # Adjust if not enough
            if n_voice + n_music < n_real_needed and len(real_pool) >= n_real_needed:
                extra = real_pool.sample(n_real_needed - n_voice - n_music, random_state=random_state)
                val_b_real = pd.concat([voice_real_pool.sample(n_voice, random_state=random_state), music_real_pool.sample(n_music, random_state=random_state), extra])
            else:
                val_b_real = pd.concat([voice_real_pool.sample(n_voice, random_state=random_state), music_real_pool.sample(n_music, random_state=random_state)])
        elif len(real_pool)>0:
            val_b_real = real_pool.sample(min(n_real_needed, len(real_pool)), random_state=random_state)
        else:
            val_b_real = pd.DataFrame(columns=df.columns)
        if len(val_b_real)>0:
            val_b = pd.concat([unseen_df, val_b_real], ignore_index=True)
            # Shuffle
            val_b = val_b.sample(frac=1, random_state=random_state).reset_index(drop=True)
        else:
            val_b = unseen_df
        # Ensure val_b size not huge: cap at 2*val_a
        if len(val_b) > len(val_a)*2 and len(val_a)>0:
            val_b = val_b.sample(len(val_a)*2, random_state=random_state).reset_index(drop=True)
        print(f"VAL-B constructed: unseen fake {n_fake} + real {len(val_b)-n_fake} = {len(val_b)} (generators {unseen_gens})")
    else:
        val_b = pd.DataFrame(columns=df.columns)
        print("Warning: no unseen generators found, VAL-B will be empty (need more datasets)")

    # Assert positive/negative sufficient for each val set
    for name, d in [("VAL-A", val_a), ("VAL-B", val_b)]:
        if len(d)==0:
            continue
        # file_fake
        n_pos = (d["file_fake"]==1).sum()
        n_neg = (d["file_fake"]==0).sum()
        print(f"{name} file_fake pos {n_pos} neg {n_neg} ({n_pos/len(d):.2f})")
        assert n_pos >= 10 and n_neg >= 10, f"{name} file_fake insufficient pos {n_pos} neg {n_neg}"
        # voice_fake
        n_pos_v = (d["voice_fake"]==1).sum()
        n_neg_v = (d["voice_fake"]==0).sum()
        print(f"{name} voice_fake pos {n_pos_v} neg {n_neg_v}")
        # music_fake
        n_pos_m = (d["music_fake"]==1).sum()
        n_neg_m = (d["music_fake"]==0).sum()
        print(f"{name} music_fake pos {n_pos_m} neg {n_neg_m}")
        # For VAL-B, we require at least one of voice or music fake to have positives; but ideally both
        # Assert at least one fake type has positives
        assert n_pos_v >= 5 or n_pos_m >= 5, f"{name} no fake positives"
        # Also assert voice_present/music_present both have pos/neg? But for now check file_fake
        # Train/VAL-B original_id overlap already handled, assert it
    # Verify train/VAL-B original_id overlap ==0
    if len(train_df)>0 and len(val_b)>0:
        # Check both original_id and utterance group
        overlap_ids = set(train_df["original_id"]) & set(val_b["original_id"])
        print(f"train/VAL-B original_id overlap {len(overlap_ids)}")
        assert len(overlap_ids)==0, f"train/VAL-B original_id overlap {len(overlap_ids)} {list(overlap_ids)[:3]}"
        overlap_utt = set(train_df.apply(_get_utterance_group, axis=1)) & set(val_b.apply(_get_utterance_group, axis=1))
        print(f"train/VAL-B utterance group overlap {len(overlap_utt)}")
        assert len(overlap_utt)==0, f"train/VAL-B utterance overlap {len(overlap_utt)} {list(overlap_utt)[:3]}"
        # Also speaker overlap
        overlap_speaker = set(train_df["speaker_id"]) & set(val_b["speaker_id"])
        print(f"train/VAL-B speaker_id overlap {len(overlap_speaker)}")
        # For strict, we allow some speaker overlap if not voice? But ideally 0; warn if >0
        if len(overlap_speaker)>0:
            print(f"Warning: train/VAL-B speaker overlap {len(overlap_speaker)}")

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
    # VAL-C/D also need asserts for class balance (they are same as VAL-A, so ok)
    for name, d in [("VAL-C", val_c), ("VAL-D", val_d)]:
        if len(d)>0:
            n_pos = (d["file_fake"]==1).sum()
            n_neg = (d["file_fake"]==0).sum()
            print(f"{name} file_fake pos {n_pos} neg {n_neg}")
            assert n_pos >= 10 and n_neg >= 10, f"{name} insufficient"

    # Save
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(pathlib.Path(out_dir)/"train.csv", index=False)
    val_a.to_csv(pathlib.Path(out_dir)/"val_a.csv", index=False)
    val_b.to_csv(pathlib.Path(out_dir)/"val_b.csv", index=False)
    val_c.to_csv(pathlib.Path(out_dir)/"val_c.csv", index=False)
    val_d.to_csv(pathlib.Path(out_dir)/"val_d.csv", index=False)
    print(f"Saved splits: train {len(train_df)} val_a {len(val_a)} val_b {len(val_b)} val_c {len(val_c)} val_d {len(val_d)} to {out_dir}")
    # Print class distributions
    for name, d in [("train", train_df), ("VAL-A", val_a), ("VAL-B", val_b), ("VAL-C", val_c), ("VAL-D", val_d)]:
        if len(d)>0:
            print(f"{name} {len(d)} file_fake {(d['file_fake']==1).sum()}/{(d['file_fake']==0).sum()} voice_fake {(d['voice_fake']==1).sum()}/{(d['voice_fake']==0).sum()} music_fake {(d['music_fake']==1).sum()}/{(d['music_fake']==0).sum()}")
    return {"train":train_df, "val_a":val_a, "val_b":val_b, "val_c":val_c, "val_d":val_d}


# This definition intentionally supersedes the legacy implementation above.  It
# keeps every original recording in exactly one split, and only then creates
# mixtures inside that split.
def build_val_sets(df, out_dir="data/splits", random_state=42):
    originals = df[~df["path"].astype(str).str.startswith("MIX::")].copy().reset_index(drop=True)
    voice_fake_gens = set(originals.loc[originals["voice_fake"] == 1, "generator"].astype(str))
    music_fake_gens = set(originals.loc[originals["music_fake"] == 1, "generator"].astype(str))

    voice_preferences = ("WF7", "ASV2021", "MLAAD")
    music_preferences = ("AudioLDM2", "MusicGen")
    holdout_voice = next((g for g in voice_preferences if g in voice_fake_gens), None)
    holdout_music = next((g for g in music_preferences if g in music_fake_gens), None)
    if holdout_voice is None or holdout_music is None:
        raise ValueError(
            "Explicit unseen-generator validation requires a known voice generator "
            f"{voice_preferences} and music generator {music_preferences}; found "
            f"voice={sorted(voice_fake_gens)}, music={sorted(music_fake_gens)}")

    held = originals[
        ((originals["voice_fake"] == 1) & (originals["generator"].astype(str) == holdout_voice)) |
        ((originals["music_fake"] == 1) & (originals["generator"].astype(str) == holdout_music))
    ].copy()
    held_groups = set(held.apply(_get_utterance_group, axis=1))
    remaining = originals[~originals.apply(lambda r: _get_utterance_group(r) in held_groups, axis=1)].copy()

    # Retry deterministic group splits until every official conditional metric
    # has both labels.  This prevents silent NaNs or deceptively easy validation.
    chosen = None
    for attempt in range(100):
        train_base, validation_pool = leakage_safe_split(
            remaining, test_size=0.30, random_state=random_state + attempt)
        val_a_base, val_b_real = leakage_safe_split(
            validation_pool, test_size=0.45, random_state=random_state + 1000 + attempt)
        val_b_base = pd.concat([held, val_b_real], ignore_index=True).drop_duplicates("original_id")
        candidates = (train_base, val_a_base, val_b_base)
        valid = all(
            len(d) and d["file_fake"].nunique() == 2
            and d.loc[d["voice_present"] == 1, "voice_fake"].nunique() == 2
            and d.loc[d["music_present"] == 1, "music_fake"].nunique() == 2
            for d in candidates)
        if valid:
            chosen = candidates
            break
    if chosen is None:
        raise RuntimeError("Could not create leakage-safe train/VAL-A/VAL-B with all official metric classes")
    train_base, val_a_base, val_b_base = chosen

    # The unseen fake generators must never appear in training.
    assert not (((train_base["voice_fake"] == 1) & (train_base["generator"] == holdout_voice)).any())
    assert not (((train_base["music_fake"] == 1) & (train_base["generator"] == holdout_music)).any())
    assert_no_base_source_overlap(train_base, val_a_base, ("train", "VAL-A"))
    assert_no_base_source_overlap(train_base, val_b_base, ("train", "VAL-B"))
    assert_no_base_source_overlap(val_a_base, val_b_base, ("VAL-A", "VAL-B"))

    train_df = add_split_internal_mixes(train_base, mixes_per_class=80, random_state=random_state)
    val_a = add_split_internal_mixes(val_a_base, mixes_per_class=20, random_state=random_state + 1)
    val_b = add_split_internal_mixes(val_b_base, mixes_per_class=20, random_state=random_state + 2)
    for d in (train_df, val_a, val_b):
        d["augment"] = "none"
    val_c, val_d = val_a.copy(), val_a.copy()
    val_c["augment"] = "codec_mp3"
    val_d["augment"] = "telephone"

    assert_no_base_source_overlap(train_df, val_a, ("train", "VAL-A"))
    assert_no_base_source_overlap(train_df, val_b, ("train", "VAL-B"))
    for name, d in (("VAL-A", val_a), ("VAL-B", val_b), ("VAL-C", val_c), ("VAL-D", val_d)):
        assert d["file_fake"].nunique() == 2, f"{name}: file EER needs real and fake"
        voice = d[d["voice_present"] == 1]
        music = d[d["music_present"] == 1]
        assert voice["voice_fake"].nunique() == 2, f"{name}: voice EER/AUC needs both labels"
        assert music["music_fake"].nunique() == 2, f"{name}: music EER/AUC needs both labels"

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_df, "val_a": val_a, "val_b": val_b, "val_c": val_c, "val_d": val_d}
    for name, split in splits.items():
        split.to_csv(out / f"{name}.csv", index=False)
    print(f"Explicit unseen generators: voice={holdout_voice}, music={holdout_music}")
    print("Saved leakage-safe post-split mixes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return splits

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
        self.separator_type="none"
        if use_demucs:
            try:
                from .models.demucs_wrapper import get_separator, HTDemucsSeparator
                self.separator=get_separator(device=device, verbose=True, use_demucs=True)
                # Check if actually loaded HTDemucs vs fallback
                if self.separator is not None and getattr(self.separator, 'use_demucs', False):
                    self.separator_type="htdemucs"
                    print(f"AudioDataset task={task} using HTDemucs separator on {device}")
                else:
                    # HTDemucs not loaded but use_demucs requested -> fail
                    raise RuntimeError("HTDemucs not available (demucs package not installed or model failed to load) but --use_demucs was specified. Install demucs or run without --use_demucs.")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Failed to initialize HTDemucs separator for task={task} use_demucs=True: {e}. Install demucs or remove --use_demucs flag.")

    def __len__(self): return len(self.df)

    def _load_and_separate(self, path_str, row):
        # Handle MIX::voice|music
        if path_str.startswith("MIX::"):
            _, rest = path_str.split("MIX::",1)
            v_path, m_path = rest.split("|",1)
            # load both
            v_wave,_ = load_audio(v_path, target_sr=self.sr)
            m_wave,_ = load_audio(m_path, target_sr=self.sr)
            snr = float(row.get("mix_snr_db", 0.0))
            if self.is_training:
                snr += random.uniform(-2.0, 2.0)
            mix = render_mixed_wave(
                v_wave, m_wave, mode=str(row.get("mix_mode", "simultaneous")),
                snr_db=snr, crossfade_sec=float(row.get("mix_crossfade_sec", 0.25)), sr=self.sr)
            # for HTDemucs path, we still need original mix; separation will be done after
            # but for task-specific, we might want to return appropriate stem:
            # For now, return mix as original; stem separation will handle
            wave=mix
            # also need to handle vocals/music for task: for mix, vocals stem is v_wave, music stem is m_wave approximated?
            # For efficiency, we can consider separated stems as v_wave/m_wave directly without demucs
            # If task is voice, return v_wave; if music, return m_wave (bypass demucs)
            if self.task=="voice":
                wave=v_wave.astype(np.float32)
            elif self.task=="music":
                wave=m_wave.astype(np.float32)
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
            wave=self._load_and_separate(path_str, row)
        else:
            wave,_ = load_audio(path_str, target_sr=self.sr)
            # HTDemucs stem separation - mandatory if use_demucs=True
            if self.use_demucs:
                if path_str.startswith("MIX::"):
                    # MIX already handled in _load_and_separate, skip HTDemucs
                    pass
                else:
                    if self.separator is None or getattr(self.separator, 'use_demucs', False)==False:
                        raise RuntimeError(f"HTDemucs separator not available for task={self.task} but --use_demucs specified (separator_type={self.separator_type})")
                    try:
                        vocals, music = self.separator.separate(wave, sr=self.sr)
                        if self.task=="voice":
                            wave=vocals
                        elif self.task=="music":
                            wave=music
                        # for multitask/file keep original
                    except Exception as e:
                        raise RuntimeError(f"HTDemucs separation failed for {path_str}: {e}")
        # Validation channel simulations apply equally to original and mixed files.
        augment = str(row.get("augment", "none")).lower()
        if augment in ("codec_mp3", "codec"):
            wave = apply_codec_sim(wave, sr=self.sr)
        elif augment in ("telephone", "tel"):
            wave = apply_telephone_sim(wave, sr=self.sr)

        if self.is_training:
            # Apply the same family of channel perturbations to every source so
            # dataset identity is a less useful shortcut than synthesis traces.
            from .augment import AugmentationPipeline
            wave = AugmentationPipeline(sr=self.sr, is_training=True)(wave)

        # segment
        seg_len=int(self.seg_sec*self.sr)
        if len(wave) < seg_len:
            wave=np.pad(wave,(0,seg_len-len(wave)))
        elif self.is_training and len(wave) > seg_len:
            s=random.randint(0,len(wave)-seg_len)
            wave=wave[s:s+seg_len]
        else:
            # eval: center crop
            if len(wave) > seg_len:
                start=(len(wave)-seg_len)//2
                wave=wave[start:start+seg_len]
        # labels
        labels = torch.tensor([row.get(k,0) for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dtype=torch.float32)
        # For task-specific, we may mask irrelevant labels? Keep all but trainer will select
        return torch.from_numpy(wave).float(), labels, str(path_str)
