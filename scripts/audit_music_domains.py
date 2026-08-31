#!/usr/bin/env python3
"""Create source-level music holdouts only when real domains are sufficient."""
import argparse
import json
import pathlib
import sys

import pandas as pd

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset import ensure_split_group_id, assert_no_base_source_overlap


def build_music_domain_holdouts(frame):
    originals=ensure_split_group_id(
        frame[~frame["path"].astype(str).str.startswith("MIX::")].copy())
    music=originals[originals["music_present"]==1].copy()
    real_sources=sorted(music.loc[music["music_fake"]==0,"source"].astype(str).unique())
    fake_generators=sorted(music.loc[music["music_fake"]==1,"generator"].astype(str).unique())
    status={"real_sources":real_sources,"fake_generators":fake_generators,
            "minimum_required":{"real_sources":2,"fake_generators":4}}
    if len(real_sources)<2 or len(fake_generators)<4:
        status.update({"status":"NOT RUN","reason":"insufficient independent music domains"})
        return status,{}
    holdouts={}
    pairs=((real_sources[-1],fake_generators[-1]),(real_sources[0],fake_generators[0]))
    for name,(real_source,fake_generator) in zip(("music_domain_holdout_a","music_domain_holdout_b"),pairs):
        test=music[((music.music_fake==0)&(music.source.astype(str)==real_source))|
                   ((music.music_fake==1)&(music.generator.astype(str)==fake_generator))].copy()
        train=music[~music.split_group_id.isin(set(test.split_group_id))].copy()
        assert_no_base_source_overlap(train,test,(name+"_train",name+"_test"))
        holdouts[name+"_train"]=train
        holdouts[name+"_test"]=test
    status["status"]="READY"
    status["rows"]={name:int(len(value)) for name,value in holdouts.items()}
    return status,holdouts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",default="data/manifest.csv")
    parser.add_argument("--output_dir",default="data/splits")
    args=parser.parse_args()
    status,holdouts=build_music_domain_holdouts(pd.read_csv(ROOT/args.manifest))
    out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    for name,frame in holdouts.items():frame.to_csv(out/(name+".csv"),index=False)
    report=ROOT/"experiments/music_domain_holdout_status.json"
    report.write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2))


if __name__=="__main__":main()
