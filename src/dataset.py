"""
Real dataset pipeline for DACON 236749 - speaker/source/generator leakage-safe, VAL-A/B/C/D, HTDemucs stems.
No synthetic fallback in final training path (synthetic only for emergency testing, not used in run_all_stages).
"""
import pathlib, random, os, hashlib, re
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

PROVENANCE_COLUMNS = (
    "dataset_name", "source_url", "version", "license",
    "allowed_for_competition", "redistribution_allowed",
    "commercial_restriction", "original_id", "speaker_id", "generator",
    "content_hash", "near_duplicate_group", "split_group_id",
)


def filter_manifest_provenance(frame, require_approved=False):
    """Validate provenance and optionally retain only explicitly approved rows."""
    missing = [column for column in PROVENANCE_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Manifest is missing provenance columns {missing}; run "
            "scripts/enrich_manifest_provenance.py")
    if frame[list(PROVENANCE_COLUMNS)].isna().any().any():
        raise RuntimeError("Manifest contains empty required provenance values")
    if not require_approved:
        return frame
    approved = frame[frame["allowed_for_competition"].astype(str).eq("YES")].copy()
    if approved.empty:
        raise RuntimeError("No explicitly competition-approved training rows remain")
    return approved.reset_index(drop=True)


def scan_real_datasets(data_root="data/raw", manifest_path="data/manifest.csv",
                       allow_path_label_fallback=False,
                       require_approved_provenance=False):
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
                    df_existing = ensure_split_group_id(df_existing)
                    # Persist grouping for the complete manifest. Approved-only
                    # filtering is an in-memory training view and must never
                    # erase rows that still require legal review.
                    df_existing.to_csv(mp, index=False)
                    return filter_manifest_provenance(
                        df_existing, require_approved_provenance)
                else:
                    missing = (~mask).sum()
                    if not allow_path_label_fallback:
                        raise RuntimeError(
                            f"Manifest {mp} has {missing} missing files; path-label "
                            "fallback is disabled")
                    print(f"Manifest {mp} has {missing} missing files, rescanning and merging")
                    # Keep existing rows that exist, rescan missing?
                    df_existing = df_existing[mask].reset_index(drop=True)
                    # Continue to scan for additional files not in manifest
                    existing_paths = set(df_existing["path"].tolist())
                # Fall through to scan additional files, but keep official labels for existing
            else:
                if not allow_path_label_fallback:
                    raise RuntimeError(f"Manifest {mp} lacks required HF metadata")
                print(f"Manifest {mp} exists but no hf_id column; explicit fallback enabled")
                existing_paths = set()
                df_existing = pd.DataFrame()
        except RuntimeError:
            raise
        except Exception as e:
            if not allow_path_label_fallback:
                raise RuntimeError(f"Failed to load mandatory manifest {mp}: {e}") from e
            print(f"Failed to load manifest {mp}: {e}, rescanning")
            existing_paths = set()
            df_existing = pd.DataFrame()
    else:
        existing_paths = set()
        df_existing = pd.DataFrame()

    if not allow_path_label_fallback:
        raise RuntimeError(
            "Refusing to infer labels from file paths. Create a metadata-backed "
            "manifest or explicitly set allow_path_label_fallback=True for diagnostics.")
    # Explicit diagnostic-only scan for files not in the manifest.
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
MIX_SNRS_DB = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
MIX_OVERLAP_FRACTIONS = (0.10, 0.25, 0.50, 0.75)
MIX_GAPS_SEC = (0.0, 0.1, 0.5, 1.0)
MIX_CROSSFADE_SEC = (0.05, 0.10, 0.25, 0.50)
MIX_GAINS_DB = (-6.0, -3.0, 0.0, 3.0)
PARTIAL_FAKE_RATIOS = (0.10, 0.20, 0.30, 0.50, 0.70)
V7_PARTIAL_FAKE_RATIOS = (0.125, 0.25, 0.50, 0.75)
V7_PARTIAL_FAKE_POSITIONS = ("start", "middle", "end")


def mixed_labels(voice_fake, music_fake):
    """Return official [file, voice, music, voice_present, music_present] labels."""
    vf, mf = int(voice_fake), int(music_fake)
    return [int(vf or mf), vf, mf, 1, 1]


def partial_fake_labels(component):
    if component == "voice":
        return [1,1,0,1,0]
    if component == "music":
        return [1,0,1,0,1]
    raise ValueError(f"Unknown partial-fake component: {component}")


def render_partial_fake_wave(real_wave, fake_wave, fake_ratio=0.2,
                             position="middle", crossfade_sec=0.02,
                             sr=TARGET_SR):
    """Replace one contiguous region of real audio with fake audio."""
    real=np.asarray(real_wave,dtype=np.float32)
    fake=np.asarray(fake_wave,dtype=np.float32)
    if real.size==0 or fake.size==0:
        raise ValueError("partial-fake source waves must be non-empty")
    total=len(real); ratio=float(np.clip(fake_ratio,0.0,1.0))
    fake_len=max(1,min(total,int(round(total*ratio))))
    if position=="start": start=0
    elif position=="end": start=total-fake_len
    elif position=="middle": start=(total-fake_len)//2
    else: raise ValueError(f"Unknown partial-fake position: {position}")
    repeats=(fake_len+len(fake)-1)//len(fake)
    insert=np.tile(fake,repeats)[:fake_len].copy()
    out=real.copy(); fade=min(int(round(crossfade_sec*sr)),fake_len//2,start,total-(start+fake_len))
    if fade>0:
        ramp=np.linspace(0,1,fade,dtype=np.float32)
        insert[:fade]=out[start:start+fade]*(1-ramp)+insert[:fade]*ramp
        insert[-fade:]=insert[-fade:]*(1-ramp)+out[start+fake_len-fade:start+fake_len]*ramp
    out[start:start+fake_len]=insert
    return np.clip(out,-1.0,1.0).astype(np.float32)


def add_partial_fake_examples(train_df, count=0, random_state=42):
    """Create VOICE partial-fake rows from TRAIN-owned originals only.

    The caller must pass one already-separated training split. Both source
    identities are retained so cross-split audits can still reason about the
    generated row. No validation/calibration/final row is accepted.
    """
    count = int(count)
    if count <= 0:
        return train_df.copy().reset_index(drop=True)
    if "data_role" in train_df.columns:
        roles = set(train_df["data_role"].astype(str).str.lower())
        forbidden = {role for role in roles if role not in ("train", "training")}
        if forbidden:
            raise ValueError(f"partial-fake creation only accepts TRAIN rows, got {sorted(forbidden)}")
    originals = train_df[~train_df["path"].astype(str).str.startswith(("MIX::", "PARTIAL::"))].copy()
    originals = ensure_split_group_id(originals)
    real = originals[(originals["voice_present"] == 1) & (originals["voice_fake"] == 0)].reset_index(drop=True)
    fake = originals[(originals["voice_present"] == 1) & (originals["voice_fake"] == 1)].reset_index(drop=True)
    if real.empty or fake.empty:
        raise ValueError("partial-fake training requires both real and fake TRAIN voice pools")
    rng = np.random.default_rng(random_state)
    rows = []
    for index in range(count):
        real_row = real.iloc[int(rng.integers(len(real)))]
        fake_row = fake.iloc[int(rng.integers(len(fake)))]
        ratio = V7_PARTIAL_FAKE_RATIOS[index % len(V7_PARTIAL_FAKE_RATIOS)]
        position = V7_PARTIAL_FAKE_POSITIONS[(index // len(V7_PARTIAL_FAKE_RATIOS)) % len(V7_PARTIAL_FAKE_POSITIONS)]
        crossfade = (0.01, 0.02, 0.05)[index % 3]
        labels = partial_fake_labels("voice")
        real_group, fake_group = str(real_row["split_group_id"]), str(fake_row["split_group_id"])
        rows.append({
            "path": f"PARTIAL::voice::{real_row['path']}|{fake_row['path']}",
            "file_fake": labels[0], "voice_fake": labels[1], "music_fake": labels[2],
            "voice_present": labels[3], "music_present": labels[4],
            "speaker_id": f"partial::{real_row['speaker_id']}::{fake_row['speaker_id']}",
            "generator": f"partial::{fake_row['generator']}",
            "source": "train_partial_voice", "dataset": "partial_voice",
            "hf_id": "generated_after_train_split",
            "original_id": f"partial::{real_row['original_id']}::{fake_row['original_id']}::{index}",
            "base_voice_id": real_group, "base_fake_voice_id": fake_group,
            "split_group_id": f"partial::{real_group}::{fake_group}",
            "partial_fake_ratio": ratio, "partial_fake_position": position,
            "partial_crossfade_sec": crossfade, "augment": "none", "data_role": "train",
        })
    result = train_df.copy()
    if "data_role" not in result.columns:
        result["data_role"] = "train"
    return pd.concat([result, pd.DataFrame(rows)], ignore_index=True, sort=False)


def derive_split_group_id(row):
    """Stable original-identity group used by every train/validation split."""
    explicit = str(row.get("split_group_id", ""))
    if explicit and explicit.lower() != "nan":
        return explicit
    source = str(row.get("source", row.get("dataset", ""))).lower()
    original_id = str(row.get("original_id", pathlib.Path(str(row.get("path", ""))).stem))
    speaker_id = str(row.get("speaker_id", "unknown"))
    wavefake_match = re.search(r"LJ\d+-\d+", original_id, flags=re.IGNORECASE)
    if "wavefake" in source or wavefake_match:
        identity = wavefake_match.group(0).upper() if wavefake_match else original_id
        return f"wavefake::{identity}"
    if "libri" in source:
        return f"libri::{speaker_id}"
    if "asvspoof" in source:
        return f"asvspoof::{speaker_id}"
    if "gtzan" in source or "fma" in source:
        return f"gtzan::{original_id}"
    if "fake_music" in source or "fakemusic" in source:
        # Generated pairs use the trailing source/prompt index, independent of generator.
        identity = re.sub(r"(?i)(musicgen|audioldm2)", "generator", original_id)
        return f"generated_music::{identity}"
    return f"{source or 'unknown'}::{original_id}"


def ensure_split_group_id(df):
    result = df.copy()
    result["split_group_id"] = result.apply(derive_split_group_id, axis=1)
    return result


def _row_source_ids(row):
    """All original recordings represented by an original or generated mix row."""
    ids = set()
    for key in ("base_voice_id", "base_music_id"):
        value = str(row.get(key, ""))
        if value and value.lower() != "nan":
            ids.add(value)
    if not ids:
        value = str(row.get("split_group_id", row.get("original_id", "")))
        if value and value.lower() != "nan":
            ids.add(value)
    return ids


def assert_no_base_source_overlap(left, right, names=("left", "right")):
    left_ids = set().union(*(_row_source_ids(r) for _, r in left.iterrows())) if len(left) else set()
    right_ids = set().union(*(_row_source_ids(r) for _, r in right.iterrows())) if len(right) else set()
    overlap = left_ids & right_ids
    assert not overlap, f"{names[0]}/{names[1]} base-source overlap: {len(overlap)} {sorted(overlap)[:3]}"


def assert_disjoint_split_groups(splits):
    """Assert pairwise disjoint original identities across major split families."""
    names = list(splits)
    groups = {
        name: set(ensure_split_group_id(frame)["split_group_id"].astype(str))
        for name, frame in splits.items()
    }
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = groups[left] & groups[right]
            assert not overlap, f"{left}/{right} split_group_id overlap: {sorted(overlap)[:3]}"
    return {name: len(value) for name, value in groups.items()}


def add_split_internal_mixes(split_df, mixes_per_class=40, random_state=42):
    """Add balanced RR/RF/FR/FF mixes using only recordings already in one split."""
    base = split_df[~split_df["path"].astype(str).str.startswith("MIX::")].copy().reset_index(drop=True)
    base = ensure_split_group_id(base)
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
                v_original, m_original = str(v["original_id"]), str(m["original_id"])
                v_id, m_id = str(v["split_group_id"]), str(m["split_group_id"])
                overlap_fraction = MIX_OVERLAP_FRACTIONS[(i + vf + mf) % len(MIX_OVERLAP_FRACTIONS)]
                gap_sec = MIX_GAPS_SEC[(i + 2 * vf + mf) % len(MIX_GAPS_SEC)]
                crossfade_sec = MIX_CROSSFADE_SEC[(i + vf + 2 * mf) % len(MIX_CROSSFADE_SEC)]
                voice_gain_db = MIX_GAINS_DB[(i + vf) % len(MIX_GAINS_DB)]
                music_gain_db = MIX_GAINS_DB[(i + mf + 1) % len(MIX_GAINS_DB)]
                rows.append({
                    "path": f"MIX::{v['path']}|{m['path']}",
                    "file_fake": labels[0], "voice_fake": labels[1], "music_fake": labels[2],
                    "voice_present": 1, "music_present": 1,
                    "speaker_id": f"mix::{v['speaker_id']}::{m['speaker_id']}",
                    "generator": f"mix::{v['generator']}::{m['generator']}",
                    "source": "split_internal_mix", "dataset": "mix", "hf_id": "generated_after_split",
                    "original_id": f"mix::{v_original}::{m_original}::{vf}{mf}::{i}",
                    "base_voice_id": v_id, "base_music_id": m_id,
                    "split_group_id": f"mix::{v['split_group_id']}::{m['split_group_id']}",
                    "mix_mode": mode, "mix_snr_db": snr,
                    "mix_overlap_fraction": overlap_fraction, "mix_gap_sec": gap_sec,
                    "mix_crossfade_sec": crossfade_sec,
                    "mix_voice_gain_db": voice_gain_db, "mix_music_gain_db": music_gain_db,
                })
    originals = ensure_split_group_id(base)
    originals["base_voice_id"] = originals.apply(
        lambda r: str(r["split_group_id"]) if int(r["voice_present"]) else "", axis=1)
    originals["base_music_id"] = originals.apply(
        lambda r: str(r["split_group_id"]) if int(r["music_present"]) else "", axis=1)
    return pd.concat([originals, pd.DataFrame(rows)], ignore_index=True, sort=False)


def render_mixed_wave(voice_wave, music_wave, mode="simultaneous", snr_db=0.0,
                      crossfade_sec=0.25, sr=TARGET_SR, overlap_fraction=0.5,
                      gap_sec=0.0, voice_gain_db=0.0, music_gain_db=0.0):
    """Render simultaneous/sequential/overlap mixtures deterministically."""
    voice = np.asarray(voice_wave, dtype=np.float32) * (10.0 ** (float(voice_gain_db) / 20.0))
    music = np.asarray(music_wave, dtype=np.float32) * (10.0 ** (float(music_gain_db) / 20.0))
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
        out = np.concatenate([voice, np.zeros(max(0, int(round(gap_sec * sr))), np.float32), music])
    elif mode == "music_then_voice":
        out = np.concatenate([music, np.zeros(max(0, int(round(gap_sec * sr))), np.float32), voice])
    elif mode == "partial_overlap":
        overlap_fraction = float(np.clip(overlap_fraction, 0.0, 1.0))
        overlap_samples = int(round(min(len(voice), len(music)) * overlap_fraction))
        out = overlap_at(voice, music, max(0, len(voice) - overlap_samples))
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


def load_manifest_row_wave(row, sr=TARGET_SR, is_training=False, use_demucs=False,
                           task="multitask", separator=None):
    """Canonical manifest-row audio path shared by train, validation and calibration."""
    path_str = str(row["path"])
    if path_str.startswith("PARTIAL::"):
        _,component,sources=path_str.split("::",2)
        real_path,fake_path=sources.split("|",1)
        real,_=load_audio(real_path,target_sr=sr);fake,_=load_audio(fake_path,target_sr=sr)
        wave=render_partial_fake_wave(
            real,fake,float(row.get("partial_fake_ratio",0.2)),
            str(row.get("partial_fake_position","middle")),
            float(row.get("partial_crossfade_sec",0.02)),sr)
    elif path_str.startswith("MIX::"):
        voice_path, music_path = path_str.split("MIX::", 1)[1].split("|", 1)
        voice, _ = load_audio(voice_path, target_sr=sr)
        music, _ = load_audio(music_path, target_sr=sr)
        snr = float(row.get("mix_snr_db", 0.0))
        if is_training:
            snr += random.uniform(-2.0, 2.0)
        wave = render_mixed_wave(
            voice, music, str(row.get("mix_mode", "simultaneous")), snr,
            float(row.get("mix_crossfade_sec", 0.25)), sr,
            float(row.get("mix_overlap_fraction", 0.5)), float(row.get("mix_gap_sec", 0.0)),
            float(row.get("mix_voice_gain_db", 0.0)), float(row.get("mix_music_gain_db", 0.0)))
    else:
        wave, _ = load_audio(path_str, target_sr=sr)
    if use_demucs and task in ("voice", "music"):
        if separator is None or not getattr(separator, "use_demucs", False):
            raise RuntimeError(f"HTDemucs unavailable for task={task}")
        vocals, accompaniment = separator.separate(wave, sr=sr)
        wave = vocals if task == "voice" else accompaniment
    augment = str(row.get("augment", "none")).lower()
    if augment in ("codec_mp3", "codec"):
        wave = apply_codec_sim(wave, sr=sr)
    elif augment in ("telephone", "tel"):
        wave = apply_telephone_sim(wave, sr=sr)
    return np.asarray(wave, dtype=np.float32)

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

def _metric_ready(frame):
    return (
        len(frame) > 0
        and frame["file_fake"].nunique() == 2
        and frame.loc[frame["voice_present"] == 1, "voice_fake"].nunique() == 2
        and frame.loc[frame["music_present"] == 1, "music_fake"].nunique() == 2
    )


def _group_partition(frame, fractions, seed):
    """Partition complete split groups according to fractions, deterministically."""
    groups = frame[["split_group_id"]].drop_duplicates().sample(frac=1, random_state=seed)
    names = list(fractions)
    total = float(sum(fractions.values()))
    cumulative = np.cumsum([fractions[name] / total for name in names])
    assignment = {}
    count = len(groups)
    for index, group in enumerate(groups["split_group_id"]):
        position = (index + 0.5) / max(1, count)
        assignment[group] = names[int(np.searchsorted(cumulative, position, side="right"))]
    return {name: frame[frame["split_group_id"].map(assignment) == name].copy() for name in names}


def build_val_sets(df, out_dir="data/splits", random_state=42):
    """Create disjoint train/model-selection/calibration/final original splits."""
    originals = df[~df["path"].astype(str).str.startswith("MIX::")].copy().reset_index(drop=True)
    originals = ensure_split_group_id(originals)
    voice_fake_gens = set(originals.loc[originals["voice_fake"] == 1, "generator"].astype(str))
    music_fake_gens = set(originals.loc[originals["music_fake"] == 1, "generator"].astype(str))
    holdout_voice = next((g for g in ("WF7", "ASV2021", "MLAAD") if g in voice_fake_gens), None)
    holdout_music = next((g for g in ("AudioLDM2", "MusicGen") if g in music_fake_gens), None)
    if holdout_voice is None or holdout_music is None:
        raise ValueError(f"Missing explicit unseen generators: voice={sorted(voice_fake_gens)} music={sorted(music_fake_gens)}")

    held_mask = (
        ((originals["voice_fake"] == 1) & (originals["generator"].astype(str) == holdout_voice))
        | ((originals["music_fake"] == 1) & (originals["generator"].astype(str) == holdout_music))
    )
    held_groups = set(originals.loc[held_mask, "split_group_id"])
    held = originals[originals["split_group_id"].isin(held_groups)].copy()
    ordinary = originals[~originals["split_group_id"].isin(held_groups)].copy()

    chosen = None
    for attempt in range(200):
        ordinary_parts = _group_partition(
            ordinary,
            {"train": 0.65, "model_selection": 0.15, "fusion_calibration": 0.10, "final_holdout": 0.10},
            random_state + attempt,
        )
        held_parts = _group_partition(
            held,
            {"model_selection": 0.50, "fusion_calibration": 0.25, "final_holdout": 0.25},
            random_state + 1000 + attempt,
        )
        parts = {"train": ordinary_parts["train"]}
        for name in ("model_selection", "fusion_calibration", "final_holdout"):
            parts[name] = pd.concat([ordinary_parts[name], held_parts[name]], ignore_index=True)
        if all(_metric_ready(parts[name]) for name in parts):
            chosen = parts
            break
    if chosen is None:
        raise RuntimeError("Could not form four disjoint metric-complete original splits")

    assert_disjoint_split_groups(chosen)
    assert not chosen["train"]["generator"].astype(str).isin((holdout_voice, holdout_music)).any()
    train_base = chosen["train"]
    model_base = chosen["model_selection"]
    calibration_base = chosen["fusion_calibration"]
    final_base = chosen["final_holdout"]

    unseen = model_base[model_base["generator"].astype(str).isin((holdout_voice, holdout_music))]
    seen = model_base[~model_base["generator"].astype(str).isin((holdout_voice, holdout_music))]
    val_a_base = seen
    # VAL-B stays within MODEL_SELECTION and includes unseen generators plus
    # deterministic negatives/other components required by all official metrics.
    val_b_base = pd.concat([unseen, seen.sample(min(len(seen), max(80, len(unseen))), random_state=random_state)],
                           ignore_index=True).drop_duplicates("original_id")
    if not _metric_ready(val_b_base):
        val_b_base = model_base.copy()

    train_df = add_split_internal_mixes(train_base, mixes_per_class=100, random_state=random_state)
    val_a = add_split_internal_mixes(val_a_base, mixes_per_class=20, random_state=random_state + 1)
    val_b = add_split_internal_mixes(val_b_base, mixes_per_class=20, random_state=random_state + 2)
    val_c, val_d = val_a.copy(), val_a.copy()
    for split in (train_df, val_a, val_b, val_c, val_d):
        split["augment"] = "none"
    val_c["augment"] = "codec_mp3"
    val_d["augment"] = "telephone"

    calibration_parts = _group_partition(
        calibration_base, {"cal_a": 1.0, "cal_b": 1.0, "cal_c": 1.0}, random_state + 2000)
    calibration_frames = []
    for index, (fold, base) in enumerate(calibration_parts.items()):
        mixed = add_split_internal_mixes(base, mixes_per_class=30, random_state=random_state + 20 + index)
        mixed["calibration_fold"] = fold
        mixed["augment"] = "none"
        calibration_frames.append(mixed)
    fusion_calibration = pd.concat(calibration_frames, ignore_index=True)
    final_holdout = add_split_internal_mixes(final_base, mixes_per_class=40, random_state=random_state + 30)
    final_holdout["augment"] = "none"

    major_with_mixes = {
        "train": train_df,
        "model_selection": pd.concat([val_a, val_b], ignore_index=True),
        "fusion_calibration": fusion_calibration,
        "final_holdout": final_holdout,
    }
    for left, right in (("train", "model_selection"), ("train", "fusion_calibration"),
                        ("train", "final_holdout"), ("model_selection", "fusion_calibration"),
                        ("model_selection", "final_holdout"), ("fusion_calibration", "final_holdout")):
        assert_no_base_source_overlap(major_with_mixes[left], major_with_mixes[right], (left, right))

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": train_df, "val_a": val_a, "val_b": val_b, "val_c": val_c, "val_d": val_d,
        "fusion_calibration": fusion_calibration, "final_holdout": final_holdout,
    }
    for name, split in splits.items():
        split.to_csv(out / f"{name}.csv", index=False)
    print(f"Explicit unseen generators excluded from TRAIN: voice={holdout_voice}, music={holdout_music}")
    print("Saved four-way leakage-safe splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
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
        # Validation transforms must be byte-deterministic. Random noise here
        # previously made VAL-D and candidate selection change between runs.
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
    def __init__(self, df, sr=16000, seg_sec=4.0, is_training=True, use_demucs=False, task="multitask", device="cpu",
                 augmentation_profile="baseline"):
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
        self.augmentation_profile=str(augmentation_profile)
        # Validation audio and deterministic channel simulations are expensive
        # to decode/resample repeatedly. Cache the final center crop per dataset
        # instance; training remains uncached so random crops/augmentations vary.
        self.eval_cache = {} if not is_training else None
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

    def __getitem__(self, idx):
        if self.eval_cache is not None and idx in self.eval_cache:
            return self.eval_cache[idx]
        row=self.df.iloc[idx]
        path_str=str(row["path"])
        wave = load_manifest_row_wave(
            row, sr=self.sr, is_training=self.is_training, use_demucs=self.use_demucs,
            task=self.task, separator=self.separator)

        if self.is_training:
            # Apply the same family of channel perturbations to every source so
            # dataset identity is a less useful shortcut than synthesis traces.
            from .augment import AugmentationPipeline
            wave = AugmentationPipeline(
                sr=self.sr, is_training=True, profile=self.augmentation_profile)(wave)

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
        result = (torch.from_numpy(np.ascontiguousarray(wave)).float(), labels, str(path_str))
        if self.eval_cache is not None:
            self.eval_cache[idx] = result
        return result
