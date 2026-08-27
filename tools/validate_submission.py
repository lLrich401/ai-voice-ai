import zipfile, pathlib, subprocess, sys, os, tempfile, pandas as pd, numpy as np, soundfile as sf
def main(zip_path="submit.zip"):
    print(f"checking {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        print(z.namelist())
        assert "script.py" in z.namelist()
        assert "requirements.txt" in z.namelist()
        assert any(n.startswith("model/") for n in z.namelist())
    tmp=tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    import pathlib
    data_test=pathlib.Path(tmp)/"data"/"test"
    data_test.mkdir(parents=True, exist_ok=True)
    sr=16000
    for i in range(3):
        sf.write(str(data_test/f"t{i}.wav"), np.random.randn(sr*4)*0.1, sr)
    env=os.environ.copy(); env["HF_HUB_OFFLINE"]="1"
    result=subprocess.run([sys.executable,"script.py"], cwd=tmp, capture_output=True, text=True, timeout=60, env=env)
    print(result.stdout)
    assert result.returncode==0
    df=pd.read_csv(pathlib.Path(tmp)/"output"/"submission.csv")
    assert not df.isna().any().any()
    print("PASS")
if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser(); p.add_argument("zip_path", nargs="?", default="submit.zip"); a=p.parse_args(); main(a.zip_path)
