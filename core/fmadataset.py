import torch
import random
import glob
import librosa
import os
import numpy as np
from random import shuffle
from typing import List


from torch.utils.data import Dataset
from scipy.signal.windows import hamming

def load_dataset(destination_path: str) -> str:
    return ''

def get_paths(dataset_path: str) -> List[str]:
    '''
    dataset_path: path to audio dataset
    returns: list of filtered paths to audio files
    '''

    exts = ('*.wav', '*.mp3')
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(dataset_path, "**", ext), recursive=True)
    paths = [p for p in paths if os.path.getsize(p) > 100 * 1024]
    shuffle(paths)
    return paths

class FMADataset(Dataset):
    '''
    Dataset for mono-channeled normalized smoothed waveform extraction
    '''

    def __init__(self, paths, dataset_type: str, ratio=0.85, sr: int = 24_000, crop_s: int = 3):
        if dataset_type == 'train':
            self.paths = paths[:int(len(paths) * ratio)]
        else:
            self.paths = paths[int(len(paths) * ratio):]
        self.sr = sr
        self.crop_len = crop_s * sr
        self.bad = set()

    def _load(self, path):
        w, sr = librosa.load(path, sr=self.sr, mono=True)
        wav = torch.from_numpy(w.astype(np.float32))
        if len(wav.shape) < 2:
            wav = wav.unsqueeze(0)
        return wav.mean(0, keepdim=True)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        for _ in range(10):
            path = self.paths[idx]
            if path in self.bad:
                idx = random.randint(0, len(self.paths) - 1)
                continue
            try:
                wav = self._load(path)
                break
            except Exception as e:
                self.bad.add(path)
                idx = random.randint(0, len(self.paths) - 1)
        else:
            wav = torch.zeros(1, self.crop_len)

        if wav.shape[1] < self.crop_len:
            rep = self.crop_len // wav.shape[1] + 1
            wav = wav.repeat(1, rep)[:, : self.crop_len]

        st = random.randint(0, wav.shape[1] - self.crop_len)
        x  = wav[:, st : st + self.crop_len]
        return x * torch.from_numpy(hamming(x.size(0))).to(torch.float32)