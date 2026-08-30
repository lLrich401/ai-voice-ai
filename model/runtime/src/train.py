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


HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")


def masked_multitask_loss(logit_t, labels, task="multitask"):
    """Presence-aware BCE matching the official conditional component metrics."""
    targets = labels.clone()
    mask = torch.ones_like(targets)
    mask[:, 1] = labels[:, 3]  # VOICE_FAKE exists only where voice exists
    mask[:, 2] = labels[:, 4]  # MUSIC_FAKE exists only where music exists
    weights = torch.ones(5, device=labels.device, dtype=labels.dtype)
    if task == "voice":
        # A vocals-stem detector's file head represents fake voice, not fake
        # accompaniment. Do not teach it contradictory music-only targets.
        targets[:, 0] = labels[:, 1]
        mask[:, 0] = labels[:, 3]
        weights = labels.new_tensor([1.0, 1.0, 0.0, 0.5, 0.0])
    elif task == "music":
        targets[:, 0] = labels[:, 2]
        mask[:, 0] = labels[:, 4]
        weights = labels.new_tensor([1.0, 0.0, 1.0, 0.0, 0.5])
    elementwise = nn.functional.binary_cross_entropy_with_logits(logit_t, targets, reduction="none")
    effective = mask * weights.unsqueeze(0)
    return (elementwise * effective).sum() / effective.sum().clamp_min(1.0)

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
            loss = masked_multitask_loss(logit_t, labels, task=task)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        total+=loss.item()
    return total/len(loader) if len(loader)>0 else 0

@torch.no_grad()
def validate(model, loader, device):
    """Fast single-crop validation for training loop (center crop)."""
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


def checkpoint_selection_score(metrics_by_split, task="pipeline"):
    """Responsibility-aware mean/worst/unseen checkpoint objective."""
    if task == "voice":
        scores = {name: 0.8 * (1.0 - float(m["voice_eer"])) + 0.2 * float(m["voice_auc"])
                  for name, m in metrics_by_split.items()}
    elif task == "music":
        scores = {name: 0.8 * (1.0 - float(m["music_eer"])) + 0.2 * float(m["music_auc"])
                  for name, m in metrics_by_split.items()}
    else:
        scores = {name: float(metrics["score"]) for name, metrics in metrics_by_split.items()}
    if not scores:
        return float("-inf")
    mean_score = float(np.mean(list(scores.values())))
    worst_score = float(np.min(list(scores.values())))
    unseen_score = scores.get("val_b", mean_score)
    return 0.55 * mean_score + 0.25 * worst_score + 0.20 * unseen_score

@torch.no_grad()
def validate_multisegment(model, df, device, use_demucs=False, task="multitask", sr=16000, seg_sec=4.0, batch_size=None):
    """Validate with multi-segment aggregation identical to script.py inference. Batched for speed."""
    model.eval()
    from .preprocess import extract_segments, aggregate_predictions, load_audio
    from .dataset import apply_codec_sim, apply_telephone_sim
    if batch_size is None:
        batch_size = 4 if str(device)=="cpu" else 16
    separator=None
    separator_type="none"
    if use_demucs:
        try:
            from .models.demucs_wrapper import get_separator
            separator=get_separator(device=str(device), verbose=False, use_demucs=True)
            separator_type="htdemucs" if getattr(separator,'use_demucs',False) else "identity"
            if separator_type!="htdemucs":
                raise RuntimeError(f"HTDemucs not available but --use_demucs specified (type={separator_type})")
        except Exception as e:
            raise RuntimeError(f"HTDemucs required for validation task={task} but failed: {e}")
        print(f"validate_multisegment using separator {separator_type} task {task} (batched, batch={batch_size})")
    else:
        separator_type="none"
        if not hasattr(validate_multisegment, "_logged"):
            print(f"validate_multisegment separator_type={separator_type} task={task} (identity, multi-segment uniform5, batch={batch_size})")
            validate_multisegment._logged=True
    # Collect all segments and mapping
    all_segments=[]  # list of np arrays [T]
    file_indices=[]  # which file each segment belongs to
    file_labels=[]  # label per file
    file_seg_counts=[]
    # First pass: load and extract segments per file
    for idx in range(len(df)):
        row=df.iloc[idx]
        path_str=str(row["path"])
        if path_str.startswith("MIX::"):
            continue
        try:
            wave,_ = load_audio(path_str, target_sr=sr)
        except Exception as e:
            print(f"load fail {path_str}: {e}")
            continue
        augment = str(row.get("augment","none")).lower() if "augment" in row else "none"
        if augment=="codec_mp3" or augment=="codec":
            wave=apply_codec_sim(wave, sr=sr)
        elif augment=="telephone" or augment=="tel":
            wave=apply_telephone_sim(wave, sr=sr)
        if use_demucs and separator is not None:
            try:
                vocals, music = separator.separate(wave, sr=sr)
                if task=="voice":
                    wave=vocals
                elif task=="music":
                    wave=music
            except Exception as e:
                raise RuntimeError(f"HTDemucs separation failed {path_str}: {e}")
        segs = extract_segments(wave, sr=sr, seg_sec=seg_sec, strategy="uniform5")
        file_labels.append(np.array([row.get(k,0) for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dtype=np.float32))
        file_seg_counts.append(len(segs))
        for s in segs:
            all_segments.append(s)
            file_indices.append(len(file_labels)-1)
    if len(all_segments)==0:
        return {"score":0,"file_eer":0.5,"voice_eer":0.5,"music_eer":0.5,"voice_auc":0.5,"music_auc":0.5}
    # Batch inference over all segments
    all_segments_np=np.stack(all_segments)  # [N_seg, T]
    # Run model in batches
    N=all_segments_np.shape[0]
    seg_probs=[]  # list of [B,5]
    for start in range(0, N, batch_size):
        end=min(start+batch_size, N)
        batch=torch.from_numpy(all_segments_np[start:end]).float().to(device)
        logits=model(batch)
        probs=np.stack([torch.sigmoid(logits[k]).cpu().numpy() for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], axis=1)
        seg_probs.append(probs)
    seg_probs=np.concatenate(seg_probs, axis=0)  # [N_seg,5]
    # Aggregate per file
    all_t=np.stack(file_labels)  # [N_file,5]
    all_p=[]
    offset=0
    for i, cnt in enumerate(file_seg_counts):
        file_seg = seg_probs[offset:offset+cnt]  # [cnt,5]
        offset+=cnt
        agg=[]
        for j, key in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"]):
            vals=file_seg[:,j]
            if key in ["file_fake","voice_fake","music_fake"]:
                agg.append(aggregate_predictions(vals, method="topk_mean", top_k=2))
            else:
                agg.append(float(np.mean(vals)))
        all_p.append(np.array(agg))
    all_p=np.stack(all_p)
    y_true={k: all_t[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    y_pred={k: all_p[:,i] for i,k in enumerate(["file_fake","voice_fake","music_fake","voice_present","music_present"])}
    return compute_dacon_metrics(y_true, y_pred)

def optimize_fusion_weights(voice_model, music_model, val_loader, device, out_path="model/fusion_weights.json"):
    """
    Validation-based fusion weight optimization for FILE_FAKE.
    Uses actual PANNs (on ORIGINAL waveform) + detector predictions for presence, not ground truth.
    Requirement 6: PANNs runs on original, voice detector on vocals, music on music.
    """
    voice_model.eval(); music_model.eval()
    all_true=[]; voice_file=[]; music_file=[]; voice_fake=[]; music_fake=[]; voice_present_pred=[]; music_present_pred=[]
    panns_model=None
    try:
        from .models.panns import PANNsPresenceWrapper
        import pathlib as pl
        if (pl.Path("model/panns/Cnn14_mAP=0.431.pth").exists() or (pl.Path(__file__).parent.parent / "model" / "panns" / "Cnn14_mAP=0.431.pth").exists()):
            panns_model=PANNsPresenceWrapper(use_pretrained=True)
            panns_model.to(device).eval()
            print("PANNs loaded for fusion presence (on ORIGINAL waveform)")
    except Exception as e:
        print(f"PANNs not loaded for fusion: {e}")
        panns_model=None
    # For fusion we need to run voice detector on vocals, music on music, PANNs on original
    # val_loader currently provides original wave (if use_demucs=False). We need separate loaders for vocals/music
    # Try to infer dataset from val_loader
    try:
        df = getattr(val_loader.dataset, 'df', None)
        is_val_dataset = df is not None
    except:
        df=None
        is_val_dataset=False
    # If we have df, we will create stem loaders for voice/music and original for PANNs
    if df is not None:
        from torch.utils.data import DataLoader
        from .dataset import AudioDataset
        # Determine use_demucs flag from original loader (should be False for original, but we need stems)
        # Create voice/music datasets with use_demucs if needed, but for PANNs we use original
        # Check if voice/music models were trained with use_demucs
        voice_use_demucs = getattr(voice_model, 'use_demucs_flag', False) if hasattr(voice_model, 'use_demucs_flag') else False
        music_use_demucs = getattr(music_model, 'use_demucs_flag', False) if hasattr(music_model, 'use_demucs_flag') else False
        # For now, assume models expect stem if they were trained with use_demucs, but we will run them on appropriate stem
        # Create loaders
        from .preprocess import extract_segments
        # We will process file by file to ensure correct stem/PANNs separation
        with torch.inference_mode():
            for idx in range(len(df)):
                row=df.iloc[idx]
                path_str=str(row["path"])
                if path_str.startswith("MIX::"):
                    continue
                from .preprocess import load_audio
                from .dataset import apply_codec_sim, apply_telephone_sim
                wave,_ = load_audio(path_str, target_sr=16000)
                augment = str(row.get("augment","none")).lower() if "augment" in row else "none"
                if augment=="codec_mp3":
                    wave=apply_codec_sim(wave, sr=16000)
                elif augment=="telephone":
                    wave=apply_telephone_sim(wave, sr=16000)
                # Get separator if needed
                # For voice/music detectors, we need vocals/music stems
                # If models were trained with demucs, we should separate; else use original
                # Try to infer from dataset use_demucs: if voice_model was trained with demucs, we separate
                # For now, we will check loader.dataset.use_demucs as proxy for training flag
                use_demucs_flag = getattr(val_loader.dataset, 'use_demucs', False)
                if use_demucs_flag:
                    from .models.demucs_wrapper import get_separator
                    sep=get_separator(device=str(device), use_demucs=True)
                    if getattr(sep,'use_demucs',False):
                        vocals, music = sep.separate(wave, sr=16000)
                    else:
                        vocals, music = wave, wave
                else:
                    # No demucs, use original for all (unified)
                    vocals, music = wave, wave
                    # For training without demucs, task-specific still uses original; so voice/music detectors run on original
                    # That's consistent with unified mode
                # Now run models on appropriate stems
                # Voice model on vocals
                wav_v = torch.from_numpy(vocals).float().unsqueeze(0).to(device)  # [1,T] but need 4s segments? Use multi-segment like script
                # For fusion validation, we should do multi-segment aggregation same as validate_multisegment
                from .preprocess import extract_segments, aggregate_predictions
                segs_v = extract_segments(vocals, sr=16000, seg_sec=4.0, strategy="uniform5")
                segs_m = extract_segments(music, sr=16000, seg_sec=4.0, strategy="uniform5")
                segs_o = extract_segments(wave, sr=16000, seg_sec=4.0, strategy="uniform5")
                batch_v = torch.from_numpy(np.stack(segs_v)).float().to(device)
                batch_m = torch.from_numpy(np.stack(segs_m)).float().to(device)
                batch_o = torch.from_numpy(np.stack(segs_o)).float().to(device)
                v_logits = voice_model(batch_v)
                m_logits = music_model(batch_m)
                # Aggregate per file
                v_file_seg = torch.sigmoid(v_logits["file_fake"]).cpu().numpy()
                m_file_seg = torch.sigmoid(m_logits["file_fake"]).cpu().numpy()
                v_fake_seg = torch.sigmoid(v_logits["voice_fake"]).cpu().numpy()
                m_fake_seg = torch.sigmoid(m_logits["music_fake"]).cpu().numpy()
                v_present_seg = torch.sigmoid(v_logits["voice_present"]).cpu().numpy()
                m_present_seg = torch.sigmoid(m_logits["music_present"]).cpu().numpy()
                # Use topk_mean for fake/file, mean for present
                v_file = aggregate_predictions(v_file_seg, method="topk_mean", top_k=2)
                m_file = aggregate_predictions(m_file_seg, method="topk_mean", top_k=2)
                v_fake = aggregate_predictions(v_fake_seg, method="topk_mean", top_k=2)
                m_fake = aggregate_predictions(m_fake_seg, method="topk_mean", top_k=2)
                v_present = float(np.mean(v_present_seg))
                m_present = float(np.mean(m_present_seg))
                # PANNs on ORIGINAL
                if panns_model is not None:
                    try:
                        p_out = panns_model(batch_o)
                        p_v = float(torch.mean(p_out["voice_present"]).item())
                        p_m = float(torch.mean(p_out["music_present"]).item())
                        v_present = 0.6*p_v + 0.4*v_present
                        m_present = 0.6*p_m + 0.4*m_present
                    except Exception as e:
                        pass
                all_true.append(np.array([row.get(k,0) for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dtype=np.float32))
                voice_file.append(v_file); music_file.append(m_file)
                voice_fake.append(v_fake); music_fake.append(m_fake)
                voice_present_pred.append(v_present); music_present_pred.append(m_present)
        if len(all_true)==0:
            raise RuntimeError("No samples for fusion optimization")
        all_true=np.stack(all_true)
        voice_file=np.array(voice_file); music_file=np.array(music_file)
        voice_fake=np.array(voice_fake); music_fake=np.array(music_fake)
        voice_present_pred=np.array(voice_present_pred); music_present_pred=np.array(music_present_pred)
    else:
        # Fallback old path: single crop on original wave (not stem) – but we still ensure PANNs on original
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
        # all_true etc already stacked in df branch, concatenated in fallback branch
        pass
    # Common handling: ensure arrays are 1D
    if isinstance(all_true, list):
        all_true=np.concatenate(all_true)
        voice_file=np.concatenate(voice_file); music_file=np.concatenate(music_file)
        voice_fake=np.concatenate(voice_fake); music_fake=np.concatenate(music_fake)
        voice_present_pred=np.concatenate(voice_present_pred); music_present_pred=np.concatenate(music_present_pred)
    # Now all_true is [N,5] or [N,5] stacked
    file_true=all_true[:,0] if all_true.ndim==2 else all_true
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
    parser.add_argument("--seed", type=int, default=20260830)
    args=parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
    # Use smaller batch for validation on CPU for speed (AASIST batch 4 is fastest)
    val_batch = min(args.batch_size, 4) if str(device)=="cpu" else args.batch_size
    val_a_loader=DataLoader(val_a_ds, batch_size=val_batch, shuffle=False, num_workers=0)

    # Optional val_b/c/d loaders for evaluation
    val_loaders={}
    for name, df in [("val_b",val_b),("val_c",val_c),("val_d",val_d)]:
        if len(df)>0:
            ds=AudioDataset(df, sr=16000, seg_sec=args.seg_sec, is_training=False, use_demucs=args.use_demucs, task=args.task, device=str(device))
            val_loaders[name]=DataLoader(ds, batch_size=val_batch, shuffle=False, num_workers=0)

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
        epoch_metrics = {"val_a": metrics_a}
        # evaluate others
        for name, loader in val_loaders.items():
            m=validate(model, loader, device)
            epoch_metrics[name] = m
            print(f"  {name} score {m['score']:.4f} file {m['file_eer']:.3f} voice {m['voice_eer']:.3f} music {m['music_eer']:.3f}")
        scheduler.step()
        selection_score = checkpoint_selection_score(epoch_metrics, task=args.task)
        print(f"  checkpoint composite {selection_score:.4f} ({args.task} responsibility + worst split + unseen VAL-B)")
        if selection_score>best_score:
            best_score=selection_score
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": best_score,
                        "selection_score": best_score, "metrics_by_split": epoch_metrics,
                        "task": args.task, "backbone": args.backbone,
                        "model_name": type(model).__name__, "base_channels": int(args.base_channels),
                        "sample_rate": 16000, "seg_sec": float(args.seg_sec),
                        "label_heads": list(HEADS), "use_demucs": bool(args.use_demucs)}, args.save_path)
            print(f"  saved {args.save_path} composite {best_score:.4f}")
            best_path=args.save_path
            no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation on all VAL sets with best model - use multi-segment aggregation matching script.py (requirement 8)
    if best_path and pathlib.Path(best_path).exists():
        ckpt=torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best {best_path} for final val (multi-segment)")
        # Use multisegment validation for actual metrics
        try:
            for name, df in [("val_a",val_a)] + [(n, splits[n]) for n in ["val_b","val_c","val_d"] if n in splits and len(splits[n])>0]:
                m=validate_multisegment(model, df, device, use_demucs=args.use_demucs, task=args.task)
                print(f"Final {name} multisegment {m}")
        except Exception as e:
            print(f"Multisegment final val failed {e}, fallback to single-crop")
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
