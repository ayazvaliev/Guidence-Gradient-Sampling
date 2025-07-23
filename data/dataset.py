import torch
import random
import librosa
import numpy as np

from torch.utils.data import Dataset
from scipy.signal.windows import hamming
from pathlib import Path
    

class FMADataset(Dataset):
    '''
    Dataset for mono-channeled normalized smoothed waveform extraction
    '''

    def __init__(self, paths: list[str], sr: int = 22_050, crop_s: float = 5, return_name: bool = False):
        super().__init__()
        self.return_name = return_name
        self.crop_len = crop_s * sr
        self.sr = sr
        self.paths = []
        for path in paths:
            try:
                w, sr_ = librosa.load(path, sr=sr, mono=True)
                assert sr_ == self.sr, f'loaded sr={sr_}, expected sr={self.sr}'
                self.paths.append(path)
            except Exception as e:
                print(f'Unable to load audio: {str(e)}')
    
    def _load(self, path: str) -> torch.Tensor:
        w, sr_ = librosa.load(path, sr=self.sr, mono=True)
        w_tensor = torch.from_numpy(w.astype(np.float32).flatten())
        if w_tensor.size(0) < self.crop_len:
            rep = self.crop_len // w_tensor.size(0) + 1
            w_tensor = w_tensor.repeat(rep)[:, :self.crop_len]
        return w_tensor

    def _normalize(self, w_tensor: torch.Tensor) -> torch.Tensor:
        w_tensor = w_tensor / torch.max(torch.abs(w_tensor))
        return w_tensor

    def _apply_hanning(self, w_tensor: torch.Tensor) -> torch.Tensor:
        assert w_tensor.ndim == 1
        return w_tensor * torch.from_numpy(hamming(w_tensor.size(0))).to(torch.float32)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        w_tensor = self._load(path)

        start = random.randint(0, w_tensor.size(0) - self.crop_len)
        w_tensor = w_tensor[start:start + self.crop_len]
        w_tensor = self._apply_hanning(w_tensor)
        w_tensor = self._normalize(w_tensor)

        if self.return_name:
            return w_tensor.unsqueeze(0), Path(path).stem
        else:
            return w_tensor.unsqueeze(0)