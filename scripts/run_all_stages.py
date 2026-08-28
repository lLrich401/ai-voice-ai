#!/usr/bin/env python3
"""
Real-data multi-stage training for DACON 236749.
No synthetic data. Requires data/raw with LibriSpeech/WaveFake/FMA/FakeMusicCaps etc.
Pipeline:
  1) Scan real datasets -> manifest + leakage-safe splits VAL-A/B/C/D
  2) Train voice detector (AASIST, vocals stem via HTDemucs)
  3) Train music detector (SpecCNN, music stem via HTDemucs)
  4) Evaluate both on VAL-A/B/C/D
  5) Optimize fusion weights via validation
  6) Save best checkpoints + fusion weights, log experiments/results.csv with only real results
"""
import sys, pathlib, json, csv, os, random
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import torch, numpy as np
from src.dataset import scan_real_datasets, build_val_sets, AudioDataset
from src.metrics import compute_dacon_metrics
from src.train import get_model, validate
from torch.utils.data import DataLoader

def evaluate_on_splits(model, splits, device, use_demucs, task):
    results={}
    for name in ["val_a","val_b","val_c","val_d"]:
        df=splits.get(name)
        if df is None or len(df)==0:
            print(f"{name} empty, skipping")
            results[name]={"score":0,"file_eer":0.5,"voice_eer":0.5,"music_eer":0.5,"voice_auc":0.5,"music_auc":0.5}
            continue
        ds=AudioDataset(df, sr=16000, seg_sec=4.0, is_training=False, use_demucs=use_demucs, task=task, device=str(device))
        loader=DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        m=validate(model, loader, device)
        print(f"{name} score {m['score']:.4f} file_eer {m['file_eer']:.3f} voice_eer {m['voice_eer']:.3f} music_eer {m['music_eer']:.3f} voice_auc {m['voice_auc']:.3f} music_auc {m['music_auc']:.3f}")
        results[name]=m
    return results

def train_detector(task, backbone, use_demucs, data_root, manifest, splits_dir, device, epochs=5, batch_size=16, lr=1e-3):
    from src.train import train_one_epoch, validate
    import torch.nn as nn
    from torch.cuda.amp import GradScaler

    # Load splits (already built)
    train_df = pathlib.Path(splits_dir)/"train.csv"
    # manifest and splits should exist
    if not train_df.exists():
        raise FileNotFoundError(f"{train_df} not found")
    import pandas as pd
    splits={}
    for k in ["train","val_a","val_b","val_c","val_d"]:
        p=pathlib.Path(splits_dir)/f"{k}.csv"
        if p.exists():
            splits[k]=pd.read_csv(p)
        else:
            splits[k]=pd.DataFrame()

    train_ds=AudioDataset(splits["train"], sr=16000, seg_sec=4.0, is_training=True, use_demucs=use_demucs, task=task, device=str(device))
    val_a_ds=AudioDataset(splits["val_a"], sr=16000, seg_sec=4.0, is_training=False, use_demucs=use_demucs, task=task, device=str(device))
    train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_a_loader=DataLoader(val_a_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model=get_model(task, backbone=backbone, base_channels=32, device=device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler=GradScaler(enabled=(device.type=="cuda"))

    best_score=-1
    save_path=f"model/{task}_{backbone}.pt"
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    patience=5; no_improve=0
    for epoch in range(epochs):
        loss=train_one_epoch(model, train_loader, opt, device, scaler, task=task)
        metrics=validate(model, val_a_loader, device)
        print(f"[{task} {backbone}] Epoch {epoch+1}/{epochs} loss {loss:.4f} VAL-A {metrics['score']:.4f}")
        scheduler.step()
        if metrics["score"]>best_score:
            best_score=metrics["score"]
            torch.save({"model": model.state_dict(), "score": best_score, "task": task, "backbone": backbone}, save_path)
            print(f"  saved {save_path}")
            no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:
                print("Early stopping")
                break
    # Load best and evaluate on all vals
    if pathlib.Path(save_path).exists():
        ckpt=torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    results=evaluate_on_splits(model, splits, device, use_demucs, task)
    return model, save_path, results

def optimize_fusion(voice_model, music_model, splits, device, use_demucs):
    """Grid search fusion weights for FILE_FAKE using VAL-A"""
    import pandas as pd
    val_a=splits["val_a"]
    if len(val_a)==0:
        print("VAL-A empty, cannot optimize fusion")
        return {"w_voice_file":0.5, "w_music_file":0.3, "w_prob_or":0.2, "val_score":0}
    # Build loaders for original wave (multitask) to get file predictions from both models?
    # For fusion, we need predictions on same original wave (not stems). So we create dataset with task=multitask and use_demucs=False for file
    # But voice/music models were trained on stems; for fusion we should run them on stems as well.
    # We'll create two loaders: voice stem and music stem separately, but need to align.
    # Simplify: run both models on original wave (multitask) and use their file predictions
    # For true HTDemucs fusion, run voice model on vocals stem, music model on music stem of same original.
    from torch.utils.data import DataLoader
    import torch
    # Create dataset that returns original, vocals, music triple
    # For simplicity, we evaluate on original wave for both, but also compute probOR from voice_fake/music_fake
    ds=AudioDataset(val_a, sr=16000, seg_sec=4.0, is_training=False, use_demucs=False, task="multitask", device=str(device))
    loader=DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    voice_model.eval(); music_model.eval()
    all_true=[]; vf=[]; mf=[]; v_file=[]; m_file=[]
    # Need also voice/music fake from respective models on stems
    # We'll create separate stem loaders for voice/music
    voice_ds=AudioDataset(val_a, sr=16000, seg_sec=4.0, is_training=False, use_demucs=use_demucs, task="voice", device=str(device))
    music_ds=AudioDataset(val_a, sr=16000, seg_sec=4.0, is_training=False, use_demucs=use_demucs, task="music", device=str(device))
    voice_loader=DataLoader(voice_ds, batch_size=16, shuffle=False, num_workers=0)
    music_loader=DataLoader(music_ds, batch_size=16, shuffle=False, num_workers=0)
    # Collect predictions
    with torch.inference_mode():
        # Get true labels from original loader
        for _, labels, _ in loader:
            all_true.append(labels.numpy())
        all_true=np.concatenate(all_true)
        # voice predictions on vocals stem
        for wav, _, _ in voice_loader:
            wav=wav.to(device)
            logits=voice_model(wav)
            vf.append(torch.sigmoid(logits["voice_fake"]).cpu().numpy())
            v_file.append(torch.sigmoid(logits["file_fake"]).cpu().numpy())
        vf=np.concatenate(vf); v_file=np.concatenate(v_file)
        for wav, _, _ in music_loader:
            wav=wav.to(device)
            logits=music_model(wav)
            mf.append(torch.sigmoid(logits["music_fake"]).cpu().numpy())
            m_file.append(torch.sigmoid(logits["file_fake"]).cpu().numpy())
        mf=np.concatenate(mf); m_file=np.concatenate(m_file)
    file_true=all_true[:,0]
    # grid search
    best_score=-1; best_w=None
    for w_v in [0.4,0.5,0.6]:
        for w_m in [0.2,0.3,0.4]:
            for w_or in [0.1,0.2,0.3]:
                s=w_v+w_m+w_or
                wv=w_v/s; wm=w_m/s; wo=w_or/s
                prob_or=1-(1-vf)*(1-mf)
                fused=wv*v_file + wm*m_file + wo*prob_or
                fused=np.clip(fused,0.01,0.99)
                y_pred={"file_fake":fused, "voice_fake":vf, "music_fake":mf, "voice_present":all_true[:,3], "music_present":all_true[:,4]}
                y_true={"file_fake":all_true[:,0], "voice_fake":all_true[:,1], "music_fake":all_true[:,2], "voice_present":all_true[:,3], "music_present":all_true[:,4]}
                metrics=compute_dacon_metrics(y_true, y_pred)
                if metrics["score"]>best_score:
                    best_score=metrics["score"]; best_w=(wv,wm,wo)
    if best_w is None:
        best_w=(0.5,0.3,0.2)
    weights={"w_voice_file":float(best_w[0]), "w_music_file":float(best_w[1]), "w_prob_or":float(best_w[2]), "val_score":float(best_score)}
    pathlib.Path("model/fusion_weights.json").parent.mkdir(parents=True, exist_ok=True)
    with open("model/fusion_weights.json","w") as f:
        json.dump(weights,f,indent=2)
    print(f"Fusion optimized {weights}")
    return weights

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_demucs", action="store_true", help="use HTDemucs stems")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()

    device=torch.device(args.device)
    print(f"Device {device} use_demucs {args.use_demucs}")

    # 1. Scan real datasets - fail if not found (no synthetic)
    try:
        if not pathlib.Path(args.manifest).exists():
            print("Scanning real datasets...")
            df=scan_real_datasets(args.data_root, args.manifest)
        else:
            import pandas as pd
            df=pd.read_csv(args.manifest)
            if len(df)==0:
                df=scan_real_datasets(args.data_root, args.manifest)
            print(f"Manifest {args.manifest} {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Real data scan failed: {e}")
        print("Ensure data/raw contains LibriSpeech, ASVspoof, WaveFake, FMA, FakeMusicCaps etc. Run scripts/download_datasets.py")
        sys.exit(1)

    # 2. Build VAL splits
    try:
        if not (pathlib.Path(args.splits_dir)/"train.csv").exists():
            print("Building leakage-safe splits VAL-A/B/C/D...")
            splits=build_val_sets(df, out_dir=args.splits_dir)
        else:
            import pandas as pd
            splits={}
            for k in ["train","val_a","val_b","val_c","val_d"]:
                p=pathlib.Path(args.splits_dir)/f"{k}.csv"
                if p.exists():
                    splits[k]=pd.read_csv(p)
                else:
                    splits[k]=pd.DataFrame()
            print(f"Loaded splits train {len(splits['train'])} val_a {len(splits['val_a'])} val_b {len(splits['val_b'])}")
    except Exception as e:
        print(f"ERROR: build splits failed {e}")
        sys.exit(1)

    # 3. Train voice detector (AASIST, vocals stem)
    print("=== Training Voice Detector (AASIST, vocals stem) ===")
    voice_model, voice_path, voice_results = train_detector(
        task="voice", backbone="aasist", use_demucs=args.use_demucs,
        data_root=args.data_root, manifest=args.manifest, splits_dir=args.splits_dir,
        device=device, epochs=args.epochs, batch_size=args.batch_size
    )
    # 4. Train music detector (SpecCNN, music stem)
    print("=== Training Music Detector (SpecCNN, music stem) ===")
    music_model, music_path, music_results = train_detector(
        task="music", backbone="spec_cnn", use_demucs=args.use_demucs,
        data_root=args.data_root, manifest=args.manifest, splits_dir=args.splits_dir,
        device=device, epochs=args.epochs, batch_size=args.batch_size
    )

    # 5. Optimize fusion
    print("=== Optimizing Fusion Weights ===")
    # Need to recreate splits dict for fusion (load again)
    import pandas as pd
    splits={}
    for k in ["train","val_a","val_b","val_c","val_d"]:
        p=pathlib.Path(args.splits_dir)/f"{k}.csv"
        if p.exists():
            splits[k]=pd.read_csv(p)
    fusion_weights=optimize_fusion(voice_model, music_model, splits, device, args.use_demucs)

    # 6. Save best as model/best.pt for inference fallback? For final, we keep both voice/music plus fusion
    # For script.py compatibility, create model/best.pt as voice model + also save music model
    # script.py expects model/best.pt (voice) and model/music_best.pt
    import shutil
    # Ensure model/best.pt is voice detector (for script compatibility, script uses AASIST for all but we provide voice as best)
    # Also save music_best
    if pathlib.Path(voice_path).exists():
        shutil.copy(voice_path, "model/best.pt")
        print(f"Copied {voice_path} -> model/best.pt")
    if pathlib.Path(music_path).exists():
        pathlib.Path("model/music_best.pt").parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(music_path, "model/music_best.pt")
        print(f"Copied {music_path} -> model/music_best.pt")

    # 7. Log results - only real experiments
    pathlib.Path("experiments").mkdir(parents=True, exist_ok=True)
    # Clear old synthetic results
    results_path=pathlib.Path("experiments/results.csv")
    # Write fresh with only real results
    with open(results_path,"w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["experiment","task","backbone","use_demucs","file_eer","voice_eer","music_eer","voice_auc","music_auc","score","val_b_score","val_c_score","val_d_score","fusion_weights","notes"])
        # voice
        vr=voice_results
        w.writerow(["voice_aasist","voice","aasist",args.use_demucs, vr["val_a"]["file_eer"], vr["val_a"]["voice_eer"], vr["val_a"]["music_eer"], vr["val_a"]["voice_auc"], vr["val_a"]["music_auc"], vr["val_a"]["score"], vr["val_b"]["score"] if "val_b" in vr else 0, vr["val_c"]["score"] if "val_c" in vr else 0, vr["val_d"]["score"] if "val_d" in vr else 0, json.dumps(fusion_weights), "real"])
        mr=music_results
        w.writerow(["music_spec_cnn","music","spec_cnn",args.use_demucs, mr["val_a"]["file_eer"], mr["val_a"]["voice_eer"], mr["val_a"]["music_eer"], mr["val_a"]["voice_auc"], mr["val_a"]["music_auc"], mr["val_a"]["score"], mr["val_b"]["score"] if "val_b" in mr else 0, mr["val_c"]["score"] if "val_c" in mr else 0, mr["val_d"]["score"] if "val_d" in mr else 0, json.dumps(fusion_weights), "real"])
    print(f"Saved {results_path} with real results only")
    print("Done. Models: model/best.pt (voice), model/music_best.pt, model/fusion_weights.json")
    # Verify checkpoints exist
    for p in ["model/best.pt","model/music_best.pt","model/fusion_weights.json"]:
        if pathlib.Path(p).exists():
            print(f"Found {p} {pathlib.Path(p).stat().st_size} bytes")
        else:
            print(f"ERROR: {p} missing")

if __name__=="__main__":
    main()
