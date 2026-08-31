import zipfile, pathlib, subprocess, sys, os, tempfile, pandas as pd, numpy as np, soundfile as sf
def main(zip_path="submit.zip"):
    print(f"checking {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        namelist=z.namelist()
        print(namelist[:30])
        assert "script.py" in namelist, "script.py missing"
        assert "requirements.txt" in namelist, "requirements.txt missing"
        assert any(n.startswith("model/") for n in namelist), "model/ missing"
        top_level={pathlib.PurePosixPath(n).parts[0] for n in namelist if n and not n.startswith("__MACOSX/")}
        allowed={"model", "script.py", "requirements.txt"}
        assert top_level <= allowed, f"unexpected top-level entries: {sorted(top_level - allowed)}"
        assert any(n.startswith("model/runtime/src/") for n in namelist), "model/runtime/src package missing"
        # Requirement 11: verify mandatory models exist
        mandatory=[
            "model/df_arena/df_arena_1b_int8.onnx",
            "model/panns/Cnn14_16k_mAP=0.438.pth",
            "model/best.pt",
            "model/music_best.pt",
            "model/fusion_weights.json",
        ]
        for m in mandatory:
            assert m in namelist, f"mandatory {m} missing in submit.zip (requirement 11)"
            print(f"found {m}")
        # Check no substring matching in script.py (requirement 13)
        script_content=z.read("script.py").decode()
        assert "substring" not in script_content.lower() or "no substring" in script_content.lower() or "exact mapping" in script_content.lower(), "script should use exact mapping"
        # Ensure no synthetic placeholder in results.csv if present
        if "experiments/results.csv" in namelist:
            import io
            csv_content=z.read("experiments/results.csv").decode()
            assert "synthetic" not in csv_content.lower() or "real" in csv_content.lower(), "results.csv contains synthetic placeholder"
            assert "would improve" not in csv_content.lower(), "results.csv contains speculative"
            print("results.csv checked")
    tmp=tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    data_root=pathlib.Path(tmp)/"data"
    data_test=data_root/"test"
    data_test.mkdir(parents=True, exist_ok=True)
    sr=16000
    for i in range(3):
        sf.write(str(data_test/f"TEST_{i:04d}.wav"), np.random.randn(sr*4)*0.1, sr)
    # Match the official data/sample_submission.csv placement and run script.py
    # with no arguments, exactly as the evaluation server does.
    sample=data_root/"sample_submission.csv"
    import pandas as pd
    pd.DataFrame({"ID":[f"TEST_{i:04d}" for i in range(3)], "FILE_FAKE_PROB":[0.5]*3, "VOICE_FAKE_PROB":[0.5]*3, "MUSIC_FAKE_PROB":[0.5]*3, "VOICE_PRESENT_PROB":[0.5]*3, "MUSIC_PRESENT_PROB":[0.5]*3}).to_csv(sample, index=False)
    env=os.environ.copy(); env["HF_HUB_OFFLINE"]="1"; env["TRANSFORMERS_OFFLINE"]="1"
    result=subprocess.run([sys.executable,"script.py"], cwd=tmp, capture_output=True, text=True, timeout=180, env=env)
    print(result.stdout)
    print(result.stderr[:2000] if result.stderr else "")
    assert result.returncode==0, f"script.py failed {result.stderr}"
    # Verify mandatory models were actually loaded (script prints verified)
    assert "Mandatory models verified" in result.stdout, "script did not verify mandatory models"
    # Verify no 0.5 fallback without error (script should not print using HeuristicModel)
    assert "HeuristicModel" not in result.stdout, "script fell back to HeuristicModel instead of failing"
    df=pd.read_csv(pathlib.Path(tmp)/"output"/"submission.csv")
    assert not df.isna().any().any(), "NaN in submission"
    assert len(df)==3, f"expected 3 rows got {len(df)}"
    # Verify exact mapping: check IDs match sample
    sdf=pd.read_csv(sample)
    assert list(df.columns)==list(sdf.columns), f"wrong columns: {list(df.columns)}"
    assert list(df.iloc[:,0].astype(str))==list(sdf.iloc[:,0].astype(str)), "sample ID order not respected (exact mapping)"
    probabilities=df.iloc[:,1:].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), "probabilities must be finite values in [0, 1]"
    print("PASS")
if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("zip_path", nargs="?", default="submit.zip"); a=p.parse_args(); main(a.zip_path)
