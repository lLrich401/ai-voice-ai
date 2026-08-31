#!/usr/bin/env python3
"""Cross-family exact and gain/codec-tolerant audio near-duplicate audit."""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.signal import stft

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.preprocess import load_audio


def spectral_fingerprint(wave,sr=16000,bands=32,time_bins=96,max_seconds=12.0):
    """Gain-tolerant spectro-temporal fingerprint.

    Keeping a time axis avoids the false matches caused by comparing only the
    global mean/std spectrum of two unrelated songs.
    """
    wave=np.asarray(wave,dtype=np.float32)
    if len(wave)>int(max_seconds*sr):
        length=int(max_seconds*sr); starts=(0,(len(wave)-length)//2,len(wave)-length)
        wave=np.concatenate([wave[s:s+length//3] for s in starts])
    rms=float(np.sqrt(np.mean(wave**2)+1e-12))
    normalized=wave/max(rms,1e-6)
    _,_,z=stft(normalized,fs=sr,nperseg=512,noverlap=384,boundary=None)
    power=np.log1p(np.abs(z)**2)
    edges=np.linspace(0,power.shape[0],bands+1,dtype=int)
    features=[]
    for left,right in zip(edges[:-1],edges[1:]):
        block=power[left:max(left+1,right)]
        features.append(block.mean(axis=0))
    features=np.stack(features)
    old_axis=np.linspace(0.0,1.0,features.shape[1])
    new_axis=np.linspace(0.0,1.0,time_bins)
    features=np.stack([np.interp(new_axis,old_axis,row) for row in features])
    vector=np.asarray(features,dtype=np.float32).ravel()
    vector=(vector-vector.mean())/(vector.std()+1e-6)
    return vector/np.linalg.norm(vector).clip(1e-6),{
        "duration":float(len(wave)/sr),"rms":rms,
        "zcr":float(np.mean(np.abs(np.diff(np.signbit(wave)))) if len(wave)>1 else 0.0)}


def fingerprint_similarity(left,right):
    return float(np.dot(np.asarray(left),np.asarray(right)))


def raw_sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def load_families(root):
    split=root/"data/splits"
    return {
        "train":pd.read_csv(split/"train.csv"),
        "model_selection":pd.concat([pd.read_csv(split/"val_a.csv"),pd.read_csv(split/"val_b.csv")]),
        "fusion_calibration":pd.read_csv(split/"fusion_calibration.csv"),
        "final_holdout":pd.read_csv(split/"final_holdout.csv"),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--threshold",type=float,default=0.9995)
    parser.add_argument("--duration_tolerance",type=float,default=0.02)
    parser.add_argument("--max_files",type=int,default=0)
    parser.add_argument("--output",default="experiments/near_duplicate_audit.json")
    args=parser.parse_args()
    records=[]
    for family,frame in load_families(ROOT).items():
        frame=frame[~frame.path.astype(str).str.startswith(("MIX::","PARTIAL::"))].drop_duplicates("path")
        if args.max_files>0:frame=frame.head(args.max_files)
        for path_value in frame.path.astype(str):
            path=pathlib.Path(path_value)
            wave,sr=load_audio(path,target_sr=16000)
            vector,stats=spectral_fingerprint(wave,sr)
            records.append({"family":family,"path":str(path),"sha256":raw_sha256(path),
                            "vector":vector,"stats":stats})
    exact=[];near=[]
    for i,left in enumerate(records):
        for right in records[i+1:]:
            if left["family"]==right["family"]:continue
            if left["sha256"]==right["sha256"]:
                exact.append({"left":left["path"],"right":right["path"],
                              "families":[left["family"],right["family"]]})
                continue
            dl=left["stats"]["duration"];dr=right["stats"]["duration"]
            if abs(dl-dr)/max(dl,dr,1e-6)>args.duration_tolerance:continue
            similarity=fingerprint_similarity(left["vector"],right["vector"])
            if similarity>=args.threshold:
                near.append({"left":left["path"],"right":right["path"],
                             "families":[left["family"],right["family"]],
                             "similarity":similarity})
    payload={"files":len(records),"threshold":args.threshold,
             "exact_cross_family":exact,"near_cross_family":near,
             "status":"PASS" if not exact and not near else "REVIEW"}
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v if not isinstance(v,list) else len(v) for k,v in payload.items()},indent=2))


if __name__=="__main__":main()
