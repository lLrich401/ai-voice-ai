#!/usr/bin/env python3
"""OOF-only logistic FILE fusion analysis; never reads final holdout."""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import script as submission
from src.metrics import compute_dacon_metrics

HEADS=("file_fake","voice_fake","music_fake","voice_present","music_present")


def assert_calibration_only(frame):
    if "calibration_fold" not in frame or frame["calibration_fold"].nunique()<2:
        raise ValueError("meta fusion requires multiple calibration folds")
    if "split" in frame and frame["split"].astype(str).str.contains("final|holdout",case=False).any():
        raise ValueError("final holdout is forbidden for meta fusion")


def meta_matrix(frame,weights,gate):
    features=[];base=[]
    for row in frame.itertuples():
        used=float(row.vp_model)>=float(gate);df=float(row.df_primary) if used else 0.5
        row_weights=dict(weights)
        if not used:
            row_weights["w_df_voice_component"]=0.0
            row_weights["w_df_music_component"]=0.0
        output=submission.fuse_prediction_features(
            df,row.vf,row.mf,row.vfile,row.mfile,row.vp_model,row.mp_model,
            row.vp_panns,row.mp_panns,row_weights)
        _,voice,music,voice_presence,music_presence=output
        features.append([df,voice,music,voice_presence,music_presence,
                         voice*voice_presence,music*music_presence,
                         row.vfile,row.mfile,row.vfile*voice_presence,
                         row.mfile*music_presence])
        base.append(output)
    return np.asarray(features,float),np.asarray(base,float)


def oof_logistic_scores(frame,features,C=0.01,class_weight=None,seed=20260831):
    assert_calibration_only(frame)
    result=np.zeros(len(frame),dtype=float)
    for fold in sorted(frame["calibration_fold"].astype(str).unique()):
        test=frame["calibration_fold"].astype(str).eq(fold).to_numpy();train=~test
        scaler=StandardScaler().fit(features[train])
        model=LogisticRegression(C=C,class_weight=class_weight,max_iter=1000,
                                 random_state=seed).fit(
                                     scaler.transform(features[train]),
                                     frame.loc[train,"y_file_fake"])
        result[test]=model.predict_proba(scaler.transform(features[test]))[:,1]
    return result


def metrics(frame,predictions):
    return compute_dacon_metrics(
        {head:frame[f"y_{head}"].to_numpy() for head in HEADS},
        {head:predictions[:,index] for index,head in enumerate(HEADS)})


def robust_total(frame,predictions):
    totals=[]
    for fold in sorted(frame["calibration_fold"].astype(str).unique()):
        mask=frame["calibration_fold"].astype(str).eq(fold).to_numpy()
        totals.append(metrics(frame.loc[mask].reset_index(drop=True),predictions[mask])["total"])
    return float(0.7*np.mean(totals)+0.3*np.min(totals))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",default="experiments/fusion_calibration_predictions.csv")
    parser.add_argument("--weights",default="model/fusion_weights.json")
    parser.add_argument("--bootstrap",type=int,default=300)
    parser.add_argument("--output",default="experiments/file_meta_oof_report.json")
    args=parser.parse_args()
    frame=pd.read_csv(ROOT/args.cache);assert_calibration_only(frame)
    weights=json.loads((ROOT/args.weights).read_text(encoding="utf-8"))
    report={"selection_data":"FUSION_CALIBRATION only","candidates":[]}
    rng=np.random.default_rng(20260831)
    for gate in (0.8,0.7,0.5,0.3):
        features,base=meta_matrix(frame,weights,gate);base_metrics=metrics(frame,base)
        base_robust=robust_total(frame,base)
        for class_weight in (None,"balanced"):
            for C in (0.01,0.1,1.0):
                scores=oof_logistic_scores(frame,features,C,class_weight)
                predicted=base.copy();predicted[:,0]=scores
                candidate_metrics=metrics(frame,predicted);deltas=[]
                for _ in range(args.bootstrap):
                    indices=rng.integers(0,len(frame),len(frame))
                    deltas.append(metrics(frame.iloc[indices].reset_index(drop=True),predicted[indices])["total"]-
                                  metrics(frame.iloc[indices].reset_index(drop=True),base[indices])["total"])
                report["candidates"].append({
                    "gate":gate,"df_usage_fraction":float((frame.vp_model>=gate).mean()),
                    "C":C,"class_weight":class_weight,"baseline":base_metrics,
                    "baseline_robust_total":base_robust,"meta":candidate_metrics,
                    "meta_robust_total":robust_total(frame,predicted),
                    "bootstrap_win_rate":float(np.mean(np.asarray(deltas)>0)),
                    "bootstrap_delta_p05":float(np.quantile(deltas,0.05)),
                    "bootstrap_delta_p95":float(np.quantile(deltas,0.95))})
    # Adoption requires a clear paired-bootstrap majority, not best in-sample mean.
    stable=[c for c in report["candidates"] if c["bootstrap_win_rate"]>=0.8
            and c["bootstrap_delta_p05"]>=0
            and c["meta_robust_total"]>=c["baseline_robust_total"]]
    report["decision"]="ADOPT" if stable else "REJECT"
    report["reason"]=("stable OOF bootstrap improvement" if stable else
                      "no candidate passed 80% win rate with non-negative 5th percentile")
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"decision":report["decision"],"reason":report["reason"],
                      "candidate_count":len(report["candidates"])},indent=2))


if __name__=="__main__":main()
