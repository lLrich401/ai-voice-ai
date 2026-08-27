"""
DACON metric
Score = 0.9 * ADS + 0.1 * CPS
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_eer(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diffs)
    eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
    if min_idx > 0 and min_idx < len(fpr)-1:
        diff = fpr - fnr
        for i in range(len(diff)-1):
            if diff[i] * diff[i+1] < 0:
                x1, y1 = fpr[i], fnr[i]
                x2, y2 = fpr[i+1], fnr[i+1]
                df = x2 - x1
                dg = y2 - y1
                denom = df - dg
                if abs(denom) > 1e-9:
                    t = (y1 - x1) / denom
                    t = np.clip(t, 0, 1)
                    eer_interp = (x1 + t*df + y1 + t*dg)/2
                    return float(np.clip(eer_interp, 0, 1))
                break
    return float(np.clip(eer, 0, 1))

def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if len(y_true) == 0:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except:
        return 0.5

def compute_dacon_metrics(y_true_dict, y_pred_dict):
    mapping = {
        "file_fake": ["file_fake","FILE_FAKE","file_fake_prob","FILE_FAKE_PROB"],
        "voice_fake": ["voice_fake","VOICE_FAKE","voice_fake_prob","VOICE_FAKE_PROB"],
        "music_fake": ["music_fake","MUSIC_FAKE","music_fake_prob","MUSIC_FAKE_PROB"],
        "voice_present": ["voice_present","VOICE_PRESENT","voice_present_prob","VOICE_PRESENT_PROB"],
        "music_present": ["music_present","MUSIC_PRESENT","music_present_prob","MUSIC_PRESENT_PROB"],
    }
    yt = {}
    yp = {}
    for norm, candidates in mapping.items():
        found_true = None
        found_pred = None
        for c in candidates:
            if c in y_true_dict:
                found_true = y_true_dict[c]
            if c in y_pred_dict:
                found_pred = y_pred_dict[c]
        if found_true is None:
            for k,v in y_true_dict.items():
                if norm in k.lower():
                    if "present" in norm and "present" in k.lower():
                        found_true = v; break
                    elif "present" not in norm and "present" not in k.lower() and "fake" in norm:
                        found_true = v; break
        if found_pred is None:
            for k,v in y_pred_dict.items():
                if norm in k.lower():
                    found_pred = v; break
        if found_true is not None:
            yt[norm] = np.asarray(found_true)
        if found_pred is not None:
            yp[norm] = np.asarray(found_pred)
    if len(yt)==0:
        yt = y_true_dict
        yp = y_pred_dict
    file_eer = compute_eer(yt["file_fake"], yp["file_fake"]) if "file_fake" in yt and "file_fake" in yp else 0.5
    voice_eer = compute_eer(yt["voice_fake"], yp["voice_fake"]) if "voice_fake" in yt and "voice_fake" in yp else 0.5
    music_eer = compute_eer(yt["music_fake"], yp["music_fake"]) if "music_fake" in yt and "music_fake" in yp else 0.5
    voice_auc = compute_auc(yt["voice_present"], yp["voice_present"]) if "voice_present" in yt and "voice_present" in yp else 0.5
    music_auc = compute_auc(yt["music_present"], yp["music_present"]) if "music_present" in yt and "music_present" in yp else 0.5
    ads = 0.5*(1-file_eer) + 0.2*(1-voice_eer) + 0.3*(1-music_eer)
    cps = 0.5*voice_auc + 0.5*music_auc
    total = 0.9*ads + 0.1*cps
    return {
        "file_eer": float(file_eer),
        "voice_eer": float(voice_eer),
        "music_eer": float(music_eer),
        "voice_auc": float(voice_auc),
        "music_auc": float(music_auc),
        "ads": float(ads),
        "cps": float(cps),
        "score": float(total),
        "total": float(total),
    }
