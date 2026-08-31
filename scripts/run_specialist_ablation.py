#!/usr/bin/env python3
"""Reproducible multi-seed SpecCNN/AASIST specialist experiment runner."""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import torch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.train import get_model


def parameter_count(task,backbone,channels):
    model=get_model(task,backbone,channels,"cpu")
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",choices=("voice","music"),default="voice")
    parser.add_argument("--backbones",nargs="+",default=["spec_cnn","aasist"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,2026,777])
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--base_channels",type=int,default=32)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--execute",action="store_true",
                        help="actually train; omission records a reproducible plan only")
    parser.add_argument("--output",default="experiments/specialist_ablation_status.json")
    args=parser.parse_args()
    results=[]
    for backbone in args.backbones:
        for seed in args.seeds:
            record={"task":args.task,"backbone":backbone,"seed":seed,
                    "epochs":args.epochs,"batch_size":args.batch_size,
                    "base_channels":args.base_channels,
                    "parameters":parameter_count(args.task,backbone,args.base_channels)}
            if not args.execute:
                record.update(status="NOT RUN",reason="execute flag not supplied")
            elif backbone=="aasist" and args.device!="cuda":
                record.update(status="NOT RUN",reason="AASIST comparison requires CUDA")
            else:
                save=ROOT/f"model/candidates/{args.task}_{backbone}_seed{seed}.pt"
                save.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,"-m","src.train","--task",args.task,
                         "--backbone",backbone,"--seed",str(seed),"--epochs",str(args.epochs),
                         "--batch_size",str(args.batch_size),"--base_channels",str(args.base_channels),
                         "--device",args.device,"--save_path",str(save)]
                started=time.perf_counter()
                completed=subprocess.run(command,cwd=ROOT,check=False)
                record.update(status="RUN" if completed.returncode==0 else "FAILED",
                              returncode=completed.returncode,
                              runtime_seconds=time.perf_counter()-started,
                              checkpoint=str(save.relative_to(ROOT)))
                if completed.returncode==0:
                    checkpoint=torch.load(save,map_location="cpu")
                    record.update(selected_epoch=int(checkpoint["epoch"])+1,
                                  selection_score=float(checkpoint["selection_score"]),
                                  metrics_by_split=checkpoint.get("metrics_by_split",{}))
            results.append(record)
    payload={"device":args.device,"cuda_available":torch.cuda.is_available(),"runs":results}
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
