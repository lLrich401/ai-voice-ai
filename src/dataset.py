import pandas as pd, torch
from torch.utils.data import Dataset
import numpy as np
from .preprocess import load_audio
class AudioDataset(Dataset):
    def __init__(self, df, sr=16000, seg_sec=4.0, is_training=True):
        self.df=df; self.sr=sr; self.seg_sec=seg_sec; self.is_training=is_training
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        wave,_=load_audio(row["path"], target_sr=self.sr)
        seg_len=int(self.seg_sec*self.sr)
        import random
        if len(wave) < seg_len:
            wave=np.pad(wave,(0,seg_len-len(wave)))
        elif self.is_training and len(wave) > seg_len:
            s=random.randint(0,len(wave)-seg_len); wave=wave[s:s+seg_len]
        else:
            wave=wave[:seg_len]
        labels=torch.tensor([row.get(k,0) for k in ["file_fake","voice_fake","music_fake","voice_present","music_present"]], dtype=torch.float32)
        return torch.from_numpy(wave).float(), labels, str(row["path"])
