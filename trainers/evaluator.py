import torch
import numpy as np
from scipy.linalg import sqrtm
from torch import Tensor
from typing import Tuple, Any

class Evaluator:
    '''Class with evaluation tools'''

    def __init__(self, sr: float, model: Any | None = None, device: torch.DeviceObjType = torch.device('cpu')):
        '''
        model: embeddings extractor model
        sr: sampling rate of waveform batches passed as input 
        device: for used model
        '''

        if model is None:
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.model.preprocess = False
            self.model.device = device
        else:
            raise NotImplementedError()

        self.model.to(device)
        self.sr = sr

    def _preprocess(self, wf_batch):
        wf_batch = wf_batch.cpu().numpy()
        wf_batch_processed = []
        for wf in wf_batch.squeeze(1):
            wf_batch_processed.append(self.model._preprocess(wf, self.sr))
        return torch.stack(wf_batch_processed)

    def _extract_embeddings(self, wf_batch_processed):
        return np.stack([self.model(wf).cpu().numpy() for wf in wf_batch_processed])

    def _calculate_stats(self, embeddings):
        #embeddings - [batch_size, num_examples, embedding_dim]
        mus = np.mean(embeddings, axis=0)
        sigmas = []
        for i in range(embeddings.shape[1]):
            sigmas.append(np.cov(embeddings[:, i, :], rowvar=False ,bias=True))
        return mus, np.asarray(sigmas)

    def calculate_fad(self, mels1: Tensor, mels2: Tensor, eps: float = 1e-6) -> float:
        '''
        mels1: log-melspectorgram tensor of first audio batch with shape [batch_size, num_examples, num_frames, num_bands]
        mels2: log-melspectorgram tensor of second audio batch with shape [batch_size, num_examples, num_frames, num_bands]
        eps: eps bias for numerical stability

        returns: averaged FAD (Frechet Audio Distance) of mels1 and mels2

        log-melspectorgram format must be the same as input format of used embeddings extractor
        '''

        mus1, sigmas1 = self._calculate_stats(self._extract_embeddings(mels1))
        mus2, sigmas2 = self._calculate_stats(self._extract_embeddings(mels2))
        fads = []
        for (mu1, mu2, sigma1, sigma2) in zip(mus1, mus2, sigmas1, sigmas2):
            covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
            if not np.isfinite(covmean).all():
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            diff = mu1 - mu2
            fads.append(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))
        return np.mean(np.asarray(fads))
    
    def calculate_mr(self, mels1, mels2) -> float:
        '''
        mels1: log-melspectorgram tensor of first audio batch with shape [batch_size, num_examples, num_frames, num_bands]
        mels2: log-melspectorgram tensor of second audio batch with shape [batch_size, num_examples, num_frames, num_bands]

        returns: averaged MR (Melspectrogram reconsruction) of mels1 and mels2
        '''

        mse = torch.mean((mels1.view(mels1.size(0), -1) - mels2.view(mels2.size(0), -1)).pow(2).mean(dim=-1))
        return mse.item()
    
    def calculate_fad_and_mr(self, audio1: Tensor, audio2: Tensor) -> Tuple[float, float]:
        '''
        audio1: first waveform batch tensor with shape [batch_size, num_channels, num_samples]
        audio2: first waveform batch tensor with shape [batch_size, num_channels, num_samples]

        returns: pair of FAD and MR of two input audio batches
        '''

        mels1, mels2 = self._preprocess(audio1), self._preprocess(audio2)
        return self.calculate_fad(mels1, mels2), self.calculate_mr(mels1, mels2)