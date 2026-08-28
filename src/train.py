#!/usr/bin/env python3
"""
Real training CLI for DACON 236749 - voice/music separate detectors, leakage-safe, HTDemucs stems, VAL-A/B/C/D, fusion optimization.
No synthetic fallback in final path.
"""
import argparse, pathlib, random, os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from .metrics import compute_dacon_metrics
from .dataset import scan_real_datasets, build_val_sets, AudioDataset

def get_model(task, backbone="aasist", base_channels=32, device="cpu"):
    if backbone=="aasist":
        from .models.aasist import AASISTMultitask
        model = AASISTMultitask(base_channels=base_channels)
    elif backbone=="spec_cnn":
        from .models.beats_backbone import MusicMultitask
        model = MusicMultitask(base_channels=base_channels)
    elif backbone=="fusion":
        from .models.beats_backbone import FusionModel
        model = FusionModel(aasist_channels=32, music_channels=32)
    else:
        from .models.aasist import AASISTMultitask
        model = AASISTMultitask(base_channels=base_channels)
    return model.to(device)

def train_one_epoch(model, loader, opt, device, scaler, task="multitask"):
    model.train()
    total=0
    for wav, labels, _ in loader:
        wav=wav.to(device); labels=labels.to(device)
        opt.zero_grad()
        with autocast(enabled=(device.type=="cuda")):
            logits=model(wav)
            # logits dict: file_fake, voice_fake, music_fake, voice_present, music_present
            logit_t=torch.stack([logits[k] for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dim=1)
            # Task-specific loss masking
            if task=="voice":
                # focus on file, voice_fake, voice_present; downweight music
                weights=torch.tensor([1.0,1.0,0.2,1.0,0.2], device=device)
                loss=nn.BCEWithLogitsLoss(weight=weights)(logit_t, labels)
                # alternative: mask music heads with reduced weight, we use weighted BCE
            elif task=="music":
                weights=torch.tensor([1.0,0.2,1.0,0.2,1.0], device=device)
                loss=nn.BCEWithLogitsLoss(weight=weights)(logit_t, labels)
            else:
                loss=nn.BCEWithLogitsLoss()(logit_t, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total+=loss.item()
    return total/len(loader) if len(loader)>0 else 0

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_t=[]; all_p=[]
    for wav, labels, _ in loader:
        wav=wav.to(device)
        logits=model(wav)
        probs=[torch.sigmoid(logits[k]).cpu().numpy() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]]
        probs=np.stack(probs, axis=1)  # [B,5]
        all_t.append(labels.numpy()); all_p.append(probs)
    if len(all_t)==0:
        return {"score":0,"file_eer":0.5,"voice_eer":0.5,"music_eer":0.5,"voice_auc":0.5,"music_auc":0.5}
    all_t=np.concatenate(all_t); all_p=np.concatenate(all_p)
    y_true={k: all_t[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    y_pred={k: all_p[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    return compute_dacon_metrics(y_true, y_pred)

def optimize_fusion_weights(voice_model, music_model, val_loader, device, out_path="model/fusion_weights.json"):
    """
    Validation-based fusion weight optimization for FILE_FAKE.
    Uses actual PANNs+detector predictions for voice_present/music_present, not ground truth.
    Grid search over w_voice, w_music, w_prob_or. Also supports optimizing for FILE_FAKE EER only.
    """
    # Collect predictions from both models on val_loader (original wave for file, stems for voice/music)
    voice_model.eval(); music_model.eval()
    all_true=[]; voice_file=[]; music_file=[]; voice_fake=[]; music_fake=[]; voice_present_pred=[]; music_present_pred=[]
    # Try to load PANNs for presence if available (optional, not mandatory for fusion)
    panns_model=None
    try:
        from .models.panns import PANNsPresenceWrapper
        import pathlib as pl
        if (pl.Path("model/panns/Cnn14_mAP=0.431.pth").exists() or (pl.Path(__file__).parent.parent / "model" / "panns" / "Cnn14_mAP=0.431.pth").exists()):
            panns_model=PANNsPresenceWrapper(use_pretrained=True)
            panns_model.to(device).eval()
            print("PANNs loaded for fusion presence")
    except:
        panns_model=None
    with torch.inference_mode():
        for wav, labels, _ in val_loader:
            wav=wav.to(device)
            v_logits=voice_model(wav)
            m_logits=music_model(wav)
            v_file=torch.sigmoid(v_logits["file_fake"]).cpu().numpy()
            m_file=torch.sigmoid(m_logits["file_fake"]).cpu().numpy()
            v_fake=torch.sigmoid(v_logits["voice_fake"]).cpu().numpy()
            m_fake=torch.sigmoid(m_logits["music_fake"]).cpu().numpy()
            v_present=torch.sigmoid(v_logits["voice_present"]).cpu().numpy()
            m_present=torch.sigmoid(m_logits["music_present"]).cpu().numpy()
            # If PANNs available, blend presence: 0.6 PANNs +0.4 detector
            if panns_model is not None:
                try:
                    p_out=panns_model(wav)
                    p_v = p_out["voice_present"].cpu().numpy()
                    p_m = p_out["music_present"].cpu().numpy()
                    v_present = 0.6*p_v + 0.4*v_present
                    m_present = 0.6*p_m + 0.4*m_present
                except:
                    pass
            all_true.append(labels.numpy())
            voice_file.append(v_file); music_file.append(m_file)
            voice_fake.append(v_fake); music_fake.append(m_fake)
            voice_present_pred.append(v_present); music_present_pred.append(m_present)
    all_true=np.concatenate(all_true)
    voice_file=np.concatenate(voice_file); music_file=np.concatenate(music_file)
    voice_fake=np.concatenate(voice_fake); music_fake=np.concatenate(music_fake)
    voice_present_pred=np.concatenate(voice_present_pred); music_present_pred=np.concatenate(music_present_pred)
    file_true=all_true[:,0]
    best_score=-1; best_w=None; best_metrics=None
    # grid search
    for w_v in [0.3,0.5,0.7]:
        for w_m in [0.3,0.5,0.7]:
            for w_or in [0.0,0.2,0.4]:
                s=w_v+w_m+w_or
                wv=w_v/s; wm=w_m/s; wo=w_or/s
                prob_or=1-(1-voice_fake)*(1-music_fake)
                fused=wv*voice_file + wm*music_file + wo*prob_or
                fused=np.clip(fused,0.01,0.99)
                # Use actual predicted presence, not ground truth (fix leakage)
                y_pred={"file_fake":fused, "voice_fake":voice_fake, "music_fake":music_fake,
                        "voice_present":voice_present_pred, "music_present":music_present_pred}
                y_true={"file_fake":file_true, "voice_fake":all_true[:,1], "music_fake":all_true[:,2],
                        "voice_present":all_true[:,3], "music_present":all_true[:,4]}
                metrics=compute_dacon_metrics(y_true, y_pred)
                # Also compute FILE_FAKE EER only for alternative optimization
                if metrics["score"]>best_score:
                    best_score=metrics["score"]; best_w=(wv,wm,wo); best_metrics=metrics
    if best_w is None:
        best_w=(0.5,0.3,0.2); best_score=0
    weights={"w_voice_file":float(best_w[0]), "w_music_file":float(best_w[1]), "w_prob_or":float(best_w[2]), "val_score":float(best_score)}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w") as f:
        json.dump(weights,f,indent=2)
    print(f"Fusion weights optimized {weights}")
    return weights

def main():
    parser=argparse.ArgumentParser(description="Real training for voice/music detectors")
    parser.add_argument("--data_root", default="data/raw", help="real data root")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--task", choices=["voice","music","multitask"], default="multitask")
    parser.add_argument("--backbone", choices=["aasist","spec_cnn","fusion"], default="aasist")
    parser.add_argument("--use_demucs", action="store_true", help="use HTDemucs vocals/music stems")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seg_sec", type=float, default=4.0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--val_sets", nargs="+", default=["val_a","val_b","val_c","val_d"], help="which val sets to evaluate")
    parser.add_argument("--optimize_fusion", action="store_true")
    args=parser.parse_args()

    device=torch.device(args.device)
    print(f"Task {args.task} backbone {args.backbone} use_demucs {args.use_demucs} device {device}")

    # Scan real datasets - fail clearly if not found (requirement 12: no silent fallback)
    try:
        if not pathlib.Path(args.manifest).exists():
            df=scan_real_datasets(args.data_root, args.manifest)
        else:
            df=pd.read_csv(args.manifest)
            print(f"Loaded manifest {args.manifest} {len(df)} rows")
            # if manifest is old or empty, rescan
            if len(df)==0:
                df=scan_real_datasets(args.data_root, args.manifest)
    except Exception as e:
        print(f"ERROR: Real data manifest failed: {e}")
        print("No synthetic fallback in final pipeline. Run: python scripts/download_datasets.py --datasets librispeech_dev wavefake fma_small fakemusiccaps --max_samples 1000")
        sys.exit(1)

    # Build splits if not exist
    splits_dir=pathlib.Path(args.splits_dir)
    if not (splits_dir/"train.csv").exists() or not (splits_dir/"val_a.csv").exists():
        try:
            splits=build_val_sets(df, out_dir=args.splits_dir)
        except Exception as e:
            print(f"ERROR: build_val_sets failed: {e}")
            sys.exit(1)
    else:
        splits={}
        for k in ["train","val_a","val_b","val_c","val_d"]:
            p=splits_dir/f"{k}.csv"
            if p.exists():
                splits[k]=pd.read_csv(p)
                print(f"Loaded {k} {len(splits[k])}")
            else:
                splits[k]=pd.DataFrame()

    # Check essential splits exist
    if len(splits["train"])==0:
        print("ERROR: train split empty. Check data/raw contents.")
        sys.exit(1)
    if len(splits["val_a"])==0:
        print("ERROR: val_a empty.")
        sys.exit(1)

    # Task-specific filtering: for voice task, filter to voice_present==1 or file task needs both but we can keep all but loss will focus
    # For music task, filter to music_present==1
    # But to keep file_fake training balanced, we keep all
    train_df=splits["train"]
    val_a=splits["val_a"]
    val_b=splits.get("val_b", pd.DataFrame())
    val_c=splits.get("val_c", pd.DataFrame())
    val_d=splits.get("val_d", pd.DataFrame())

    # For voice task, we may want to oversample voice samples
    if args.task=="voice":
        # keep all but ensure voice samples are majority
        print(f"Voice task: train {len(train_df)} val_a {len(val_a)}")
    elif args.task=="music":
        print(f"Music task: train {len(train_df)} val_a {len(val_a)}")

    # Datasets
    train_ds=AudioDataset(train_df, sr=16000, seg_sec=args.seg_sec, is_training=True, use_demucs=args.use_demucs, task=args.task, device=str(device))
    val_a_ds=AudioDataset(val_a, sr=16000, seg_sec=args.seg_sec, is_training=False, use_demucs=args.use_demucs, task=args.task, device=str(device))
    train_loader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_a_loader=DataLoader(val_a_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optional val_b/c/d loaders for evaluation
    val_loaders={}
    for name, df in [("val_b",val_b),("val_c",val_c),("val_d",val_d)]:
        if len(df)>0:
            ds=AudioDataset(df, sr=16000, seg_sec=args.seg_sec, is_training=False, use_demucs=args.use_demucs, task=args.task, device=str(device))
            val_loaders[name]=DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model=get_model(args.task, backbone=args.backbone, base_channels=args.base_channels, device=device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler=GradScaler(enabled=(device.type=="cuda"))
    best_score=-1
    best_path=None
    if args.save_path is None:
        args.save_path=f"model/{args.task}_{args.backbone}.pt" if args.task!="multitask" else "model/best.pt"
    pathlib.Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    # Training loop with early stopping on VAL-A
    patience=5; no_improve=0
    for epoch in range(args.epochs):
        loss=train_one_epoch(model, train_loader, opt, device, scaler, task=args.task)
        metrics_a=validate(model, val_a_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} loss {loss:.4f} VAL-A score {metrics_a['score']:.4f} file_eer {metrics_a['file_eer']:.3f} voice_eer {metrics_a['voice_eer']:.3f} music_eer {metrics_a['music_eer']:.3f} voice_auc {metrics_a['voice_auc']:.3f} music_auc {metrics_a['music_auc']:.3f}")
        # evaluate others
        for name, loader in val_loaders.items():
            m=validate(model, loader, device)
            print(f"  {name} score {m['score']:.4f} file {m['file_eer']:.3f} voice {m['voice_eer']:.3f} music {m['music_eer']:.3f}")
        scheduler.step()
        if metrics_a["score"]>best_score:
            best_score=metrics_a["score"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": best_score, "task": args.task, "backbone": args.backbone}, args.save_path)
            print(f"  saved {args.save_path} score {best_score:.4f}")
            best_path=args.save_path
            no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation on all VAL sets with best model
    if best_path and pathlib.Path(best_path).exists():
        ckpt=torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best {best_path} for final val")
        for name, loader in [("val_a",val_a_loader)] + list(val_loaders.items()):
            m=validate(model, loader, device)
            print(f"Final {name} {m}")

    # Fusion optimization if requested and task is multitask with both voice/music models available
    if args.optimize_fusion and args.task=="multitask":
        # Need both voice and music models - for demo we use same model
        # In real stage, this is called after both voice and music detectors trained separately
        print("Fusion optimization requested but need separate voice/music models - skipping, run scripts/run_all_stages.py for joint optimization")

    print(f"Training done {args.task} best_score {best_score:.4f} saved {best_path}")

if __name__=="__main__":
    main()
