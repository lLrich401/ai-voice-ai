#!/usr/bin/env python3
"""
Real dataset downloader for DACON deepvoice - preserves official HF metadata/label/generator in manifest.csv
Does NOT infer labels from path strings. Each file's label is taken from original dataset metadata.
Saves audio as original bytes (flac/wav) and records manifest with official labels.
Supports: librispeech (real voice), ASVspoof (bonafide/spoof), WaveFake (ajaykarthick), GTZAN (real music), FakeMusicCaps (metadata fallback)
"""
import os, pathlib, argparse, json, hashlib, csv, sys
from pathlib import Path

DATASETS = {
    "librispeech_dev": {
        "hf_id": "openslr/librispeech_asr",
        "config": "clean",
        "split": "validation",
        "license": "CC BY 4.0",
        "url": "https://www.openslr.org/12",
        "category": "Real Voice",
        "samples": "2700 utterances (dev-clean)",
        "use": "Real Voice train/val",
    },
    "asvspoof2019": {
        "hf_id": "Bisher/ASVspoof_2019_LA",
        "split": "train",
        "license": "Custom research (ASVspoof)",
        "url": "https://datashare.ed.ac.uk/handle/10283/3336",
        "category": "Voice Fake/Real",
        "samples": "25380 (2580 bonafide, 22800 spoof)",
        "use": "Voice bonafide/spoof with attack labels",
    },
    "wavefake_ajay": {
        "hf_id": "ajaykarthick/wavefake-audio",
        "split": "train",
        "license": "MIT",
        "url": "https://huggingface.co/datasets/ajaykarthick/wavefake-audio",
        "category": "Fake Voice",
        "samples": "Real + Fake (WF1 etc)",
        "use": "WaveFake with real_or_fake",
    },
    "gtzan_real": {
        "hf_id": "sanchit-gandhi/gtzan",
        "split": "train",
        "license": "CC BY-SA",
        "url": "https://huggingface.co/datasets/sanchit-gandhi/gtzan",
        "category": "Real Music",
        "samples": "1000 tracks GTZAN (10 genres, 30s each)",
        "use": "Real Music diverse",
    },
    "fakemusiccaps": {
        "hf_id": "DeepFense/FakeMusicCaps",
        "split": "train",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/DeepFense/FakeMusicCaps",
        "category": "Music Real/Fake",
        "samples": "5590 bonafide + 5589 spoof (MusicGen/AudioLDM2) - metadata only, audio via GTZAN fallback if needed",
        "use": "MusicGen/AudioLDM2 fake (metadata)",
    },
}

def save_audio_bytes(audio_dict, out_path: Path):
    if isinstance(audio_dict, dict):
        b = audio_dict.get("bytes")
        orig_path = audio_dict.get("path", "")
        ext = Path(orig_path).suffix if orig_path else ".wav"
        if not ext:
            ext=".wav"
        out_path = out_path.with_suffix(ext)
        if b is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b)
            return str(out_path), ext
        else:
            arr = audio_dict.get("array")
            sr = audio_dict.get("sampling_rate", 16000)
            if arr is not None:
                import soundfile as sf
                import numpy as np
                arr = np.array(arr)
                if arr.ndim>1:
                    arr = arr.mean(axis=0)
                out_path = out_path.with_suffix(".wav")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_path), arr, sr)
                return str(out_path), ".wav"
    if isinstance(audio_dict, bytes):
        out_path = out_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_dict)
        return str(out_path), ".wav"
    raise ValueError("Unknown audio format")

def validate_content_uniqueness(paths, dataset_key, minimum_ratio=0.90):
    """Fail fast when a downloader silently repeats audio under new filenames."""
    paths = [Path(path) for path in paths]
    if not paths:
        return {"files": 0, "unique": 0, "ratio": 0.0}
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    ratio = len(digests) / len(paths)
    if ratio < float(minimum_ratio):
        raise RuntimeError(
            f"{dataset_key} content-integrity failure: {len(digests)}/{len(paths)} "
            f"unique files ({ratio:.1%}, required >= {minimum_ratio:.1%})"
        )
    return {"files": len(paths), "unique": len(digests), "ratio": ratio}

def download_hf_with_manifest(hf_id, local_dir, manifest_writer, config=None, split=None, max_samples=1000, dataset_key="unknown"):
    try:
        from datasets import load_dataset, Audio
        print(f"Downloading {hf_id} config={config} split={split} max={max_samples} ...")
        kwargs={}
        if config:
            kwargs["name"]=config
        if split:
            kwargs["split"]=split
        kwargs["streaming"]=True
        ds = load_dataset(hf_id, **kwargs)
        # Handle path-based datasets (no audio column) - we will handle separately
        is_path_based = False
        if hf_id in ["DeepFense/FakeMusicCaps", "DeepFense/WaveFake"]:
            # These are metadata only, no audio bytes - we will create manifest but need audio fallback
            # For FakeMusicCaps, we can still create manifest with labels but audio will be placeholder
            # For now, skip downloading audio for these, but still create manifest rows with path as ID and no file
            # Instead, we will not save audio files for these, but record that audio not available
            print(f"Dataset {hf_id} is metadata-only (no audio), will create manifest but audio files need external source. Skipping file save, but recording labels.")
            # For FakeMusicCaps, we can try to use it as label source but audio not available - we will skip
            # Return 0 to indicate no files, but we still want to count
            # Instead, we will iterate and create manifest rows without saving audio, using a placeholder path that points to existing GTZAN or librispeech for demo
            # For now, just skip
            print(f"Skipping {hf_id} - metadata only, use GTZAN for music instead")
            return 0

        audio_col=None
        for col in ["audio","bonafide_audio","spoof_audio","music","clip"]:
            if col in ds.features:
                audio_col=col
                break
        if audio_col:
            try:
                ds = ds.cast_column(audio_col, Audio(decode=False))
            except Exception as e:
                print(f"cast {audio_col} failed {e}")
        else:
            print(f"No audio column found in {hf_id}, features {list(ds.features.keys())}")
            return 0
        # Shuffle for balanced sampling (large buffer to handle ordered datasets like ASVspoof)
        try:
            ds = ds.shuffle(seed=42, buffer_size=10000)
        except:
            pass
        out_base = Path(local_dir) / hf_id.replace("/", "_")
        out_base.mkdir(parents=True, exist_ok=True)
        count=0
        saved_paths=[]
        pending_manifest_rows=[]
        for i, item in enumerate(ds):
            if count>=max_samples:
                break
            audio=None
            for k in ["audio","bonafide_audio","spoof_audio","music"]:
                if k in item and item[k] is not None:
                    audio=item[k]
                    break
            if audio is None:
                for k,v in item.items():
                    if isinstance(v, dict) and "bytes" in v:
                        audio=v; break
            if audio is None:
                continue
            try:
                raw_id = str(item.get("ID") or item.get("audio_id") or item.get("audio_file_name") or item.get("id") or item.get("file") or f"{i:06d}")
                # For WaveFake, include real_or_fake to avoid collision of same LJ utterance with different generators
                rf_for_name = str(item.get("real_or_fake","") or item.get("generator","") or "")
                if hf_id == "ajaykarthick/wavefake-audio" and rf_for_name:
                    raw_id = f"{raw_id}_{rf_for_name}"
                # For GTZAN, ensure track uniqueness: include genre and id
                if hf_id == "sanchit-gandhi/gtzan":
                    genre = str(item.get("genre",""))
                    raw_id = f"{raw_id}_genre{genre}"
                base_name = raw_id.replace("/","_").replace("\\","_").replace(".wav","").replace(".flac","").replace(".","_")
                # Ensure uniqueness with dataset prefix
                base_name = f"{dataset_key}_{base_name}"
                out_path = out_base / f"{base_name}"
                saved_path, ext = save_audio_bytes(audio, out_path)
            except Exception as e:
                continue

            voice_present=0; music_present=0; voice_fake=0; music_fake=0; file_fake=0
            speaker_id="unknown"; generator="unknown"; source=dataset_key
            if hf_id == "openslr/librispeech_asr":
                voice_present=1; music_present=0; voice_fake=0; music_fake=0; file_fake=0
                speaker_id=str(item.get("speaker_id", f"libri_{hashlib.md5(base_name.encode()).hexdigest()[:6]}"))
                generator="LibriSpeech"
                source="librispeech_dev"
            elif hf_id == "Bisher/ASVspoof_2019_LA":
                key=int(item.get("key", 0))
                system_id=str(item.get("system_id", "-"))
                speaker_id=str(item.get("speaker_id", "asv_unknown"))
                if key==0 and system_id=="-":
                    voice_present=1; voice_fake=0; file_fake=0
                    generator="bonafide"
                else:
                    voice_present=1; voice_fake=1; file_fake=1
                    generator=system_id if system_id!="-" else "ASV_spoof"
                music_present=0; music_fake=0
                source="asvspoof2019"
            elif hf_id == "ajaykarthick/wavefake-audio":
                rf=str(item.get("real_or_fake","")).strip()
                # WaveFake: R = bonafide/real, WF1~WF7 = fake (requirement 1)
                if rf.upper() == "R" or rf.lower() in ["real","bonafide"]:
                    voice_present=1; voice_fake=0; file_fake=0
                    generator="R"  # keep original R for audit
                elif rf.upper().startswith("WF"):
                    voice_present=1; voice_fake=1; file_fake=1
                    generator=str(rf)  # WF1~WF7
                else:
                    # Fallback: treat unknown as fake if contains WF, else real
                    if "WF" in rf.upper():
                        voice_present=1; voice_fake=1; file_fake=1
                        generator=str(rf) if rf else "WaveFake"
                    else:
                        voice_present=1; voice_fake=0; file_fake=0
                        generator=str(rf) if rf else "R"
                speaker_id=f"wavefake_{hashlib.md5(base_name.encode()).hexdigest()[:6]}"
                music_present=0; music_fake=0
                source="wavefake_ajay"
            elif hf_id == "sanchit-gandhi/gtzan":
                # GTZAN: real music, genre as generator
                genre=str(item.get("genre", 0))
                # Map genre int to name if needed
                voice_present=0; music_present=1; music_fake=0; file_fake=0
                voice_fake=0
                generator=f"GTZAN_genre_{genre}"
                speaker_id=f"gtzan_{genre}_{hashlib.md5(base_name.encode()).hexdigest()[:4]}"
                source="gtzan_real"
            elif hf_id == "DeepFense/FakeMusicCaps":
                # This is metadata only, but if we reach here, handle
                label=str(item.get("label","")).lower()
                path_str=str(item.get("path",""))
                gen=path_str.split("/")[0] if "/" in path_str else "unknown"
                if label=="bonafide":
                    voice_present=0; music_present=1; music_fake=0; file_fake=0
                    generator=gen if gen!="unknown" else "MusicCaps_real"
                else:
                    voice_present=0; music_present=1; music_fake=1; file_fake=1
                    generator=gen if gen!="unknown" else "FakeMusic"
                voice_fake=0
                speaker_id=f"fakemusic_{gen}_{hashlib.md5(base_name.encode()).hexdigest()[:4]}"
                source="fakemusiccaps"
            else:
                voice_present=1; voice_fake=0; file_fake=0
                generator="unknown"
                speaker_id=f"unk_{hashlib.md5(base_name.encode()).hexdigest()[:6]}"
            row={
                "path": saved_path,
                "file_fake": int(file_fake),
                "voice_fake": int(voice_fake),
                "music_fake": int(music_fake),
                "voice_present": int(voice_present),
                "music_present": int(music_present),
                "speaker_id": str(speaker_id),
                "generator": str(generator),
                "source": str(source),
                "dataset": str(dataset_key),
                "hf_id": hf_id,
                "original_id": base_name,
            }
            pending_manifest_rows.append(row)
            saved_paths.append(saved_path)
            count+=1
            if count%200==0:
                print(f"  saved {count}/{max_samples} ...")
        if hf_id == "sanchit-gandhi/gtzan":
            integrity = validate_content_uniqueness(saved_paths, dataset_key, minimum_ratio=0.90)
            print(f"GTZAN integrity: {integrity['unique']}/{integrity['files']} unique files")
        for row in pending_manifest_rows:
            manifest_writer.writerow(row)
        print(f"Saved {count} files to {out_base} with manifest")
        return count
    except Exception as e:
        print(f"Failed {hf_id}: {e}")
        import traceback; traceback.print_exc()
        return 0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/raw")
    parser.add_argument("--manifest", default="./data/manifest.csv")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--datasets", nargs="+", default=["librispeech_dev","asvspoof2019","wavefake_ajay","gtzan_real"])
    args=parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    manifest_path=Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames=["path","file_fake","voice_fake","music_fake","voice_present","music_present","speaker_id","generator","source","dataset","hf_id","original_id"]
    with open(manifest_path,"w", newline="", encoding="utf-8") as mf:
        writer=csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        total=0
        for name in args.datasets:
            if name not in DATASETS:
                print(f"Unknown {name}, skipping")
                continue
            info=DATASETS[name]
            print(f"\n=== {name} ===")
            print(f"HF {info.get('hf_id')} License {info.get('license')} URL {info.get('url')}")
            if info.get("hf_id") is None:
                print(f"Skipping {name} - no HF id")
                continue
            # For fakemusiccaps metadata-only, we skip file save but still need to handle
            if name=="fakemusiccaps":
                print("Skipping fakemusiccaps file save (metadata only) - use gtzan for music")
                continue
            count=download_hf_with_manifest(
                info["hf_id"], args.output, writer,
                config=info.get("config"), split=info.get("split"),
                max_samples=args.max_samples, dataset_key=name
            )
            total+=count
            mf.flush()
    print(f"\nTotal {total} files downloaded to {args.output}, manifest {manifest_path}")
    if manifest_path.exists():
        import pandas as pd
        df=pd.read_csv(manifest_path)
        print(f"Manifest rows {len(df)}")
        if len(df)>0:
            print(df.head().to_string())
            print(df["generator"].value_counts().head(10))
            print(df.groupby(["dataset","file_fake"]).size())

if __name__=="__main__":
    main()
