
#!/usr/bin/env python3
"""
Real dataset downloader for DACON deepvoice
Downloads public datasets with clear licenses
"""
import os, pathlib, argparse, json, shutil, hashlib
from pathlib import Path

# Dataset configs with URLs and licenses
DATASETS = {
    "librispeech_dev": {
        "hf_id": "openslr/librispeech_asr",
        "config": "clean",
        "split": "validation",
        "license": "CC BY 4.0",
        "url": "https://www.openslr.org/12",
        "category": "Real Voice",
        "samples": "2700 utterances (dev-clean)",
        "use": "Real Voice train/val"
    },
    "common_voice_small": {
        "hf_id": "mozilla-foundation/common_voice_11_0",
        "config": "en",
        "split": "train[:1000]",
        "license": "CC-0",
        "url": "https://commonvoice.mozilla.org/datasets",
        "category": "Real Voice",
        "samples": "1000",
        "use": "Real Voice diversity"
    },
    "asvspoof2019": {
        "hf_id": "jungjee/asvspoof2019",
        "license": "Custom research (ASVspoof)",
        "url": "https://datashare.ed.ac.uk/handle/10283/3336",
        "category": "Fake Voice",
        "samples": "25000 fake, 6000 real",
        "use": "Fake Voice TTS/VC"
    },
    "wavefake": {
        "hf_id": "hamid2/WaveFake",
        "license": "MIT",
        "url": "https://github.com/RUB-SysSec/WaveFake",
        "category": "Fake Voice",
        "samples": "117k fake (MelGAN, HiFiGAN etc)",
        "use": "Neural vocoder fake"
    },
    "mlaad_small": {
        "hf_id": "Multi-Language-Anti-Spoofing/MLAAD",
        "config": "en",
        "split": "train[:2000]",
        "license": "CC BY 4.0",
        "url": "https://github.com/Multi-Language-Anti-Spoofing/MLAAD",
        "category": "Fake Voice",
        "samples": "2000 (subset)",
        "use": "Multilingual TTS"
    },
    "fma_small": {
        "hf_id": "mdeff/fma",
        "config": "small",
        "split": "train[:1000]",
        "license": "CC BY 4.0",
        "url": "https://github.com/mdeff/fma",
        "category": "Real Music",
        "samples": "1000 tracks (small)",
        "use": "Real Music diverse"
    },
    "musiccaps": {
        "hf_id": "google/musiccaps",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/google/musiccaps",
        "category": "Real Music",
        "samples": "5500",
        "use": "Real Music captions"
    },
    "fakemusiccaps": {
        "hf_id": "nikhilchandak/fakemusiccaps",
        "license": "CC BY 4.0",
        "url": "https://huggingface.co/datasets/nikhilchandak/fakemusiccaps",
        "category": "Fake Music",
        "samples": "5500 fake + 5500 real",
        "use": "Fake Music MusicGen/AudioLDM2"
    }
}

def download_hf_dataset(hf_id, local_dir, config=None, split=None, max_samples=1000):
    try:
        from datasets import load_dataset
        print(f"Downloading {hf_id} ...")
        kwargs = {}
        if config:
            kwargs["name"] = config
        if split:
            kwargs["split"] = split
        ds = load_dataset(hf_id, **kwargs, trust_remote_code=True)
        # Take subset
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        # Save to disk
        out = Path(local_dir) / hf_id.replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        # Try to save audio files
        count=0
        for i, item in enumerate(ds):
            # Try to find audio
            audio = None
            for k in ["audio", "wav", "mp3", "flac"]:
                if k in item and item[k] is not None:
                    audio = item[k]
                    break
            if audio is None:
                continue
            # audio is dict with array and sampling_rate
            try:
                import soundfile as sf
                arr = audio["array"] if isinstance(audio, dict) else audio
                sr = audio["sampling_rate"] if isinstance(audio, dict) else 16000
                if hasattr(arr, "shape") and len(arr.shape)>1:
                    arr = arr.mean(axis=0)
                # Save as wav 16k
                import librosa
                if sr != 16000:
                    arr = librosa.resample(arr.astype(float), orig_sr=sr, target_sr=16000)
                    sr=16000
                sf.write(str(out / f"{i:06d}.wav"), arr, sr)
                count+=1
            except Exception as e:
                continue
        print(f"Saved {count} files to {out}")
        return count
    except Exception as e:
        print(f"Failed {hf_id}: {e}")
        import traceback; traceback.print_exc()
        return 0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/raw")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--datasets", nargs="+", default=["librispeech_dev","wavefake","fma_small"])
    args=parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    total=0
    for name in args.datasets:
        if name not in DATASETS:
            print(f"Unknown {name}, skipping")
            continue
        info=DATASETS[name]
        print(f"\n=== {name} ===")
        print(f"License {info['license']} URL {info['url']}")
        count=download_hf_dataset(info["hf_id"], args.output, config=info.get("config"), split=info.get("split"), max_samples=args.max_samples)
        total+=count
        # Write data_sources entry
    print(f"\nTotal {total} files downloaded to {args.output}")
    # Update docs/data_sources.md
    docs = Path("docs/data_sources.md")
    if docs.exists():
        print(f"docs exists {docs.stat().st_size}")

if __name__=="__main__":
    main()
