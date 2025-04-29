import torch
from audio_diffusion_pytorch.diffusion import Schedule, LinearSchedule, Sampler, VDiffusion, extend_dim
from torch import Tensor
from math import floor, pi
from typing import Tuple
import torch.nn as nn
from einops import repeat
from tqdm import tqdm

class VSamplerWithGradientGuidence(Sampler):
    '''Implementation of waveform DDIM gradient guidence sampling from https://arxiv.org/abs/2311.00613'''

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, L_C: float, R_C: float, SR: int, schedule: Schedule = LinearSchedule(), psi = 3 * 1e-3):
        '''
        net: V-predictive net
        L_C: Left context length ratio
        R_C: Right context length ratio
        schedule: Schedule for time steps
        psi: Gradient guidence step
        '''
        super().__init__()
        self.net = net
        self.schedule = schedule
        self.psi = psi
        self.L_C = L_C
        self.R_C = R_C
        self.SR = SR
        self.d = lambda x, y: torch.linalg.vector_norm(y - x, ord=1, dim=-1)
        self.LEFT_CTX_SIZE = floor(self.SR * self.L_C)
        self.RIGHT_CTX_SIZE = floor(self.SR * self.R_C)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    @torch.no_grad()
    def forward(  # type: ignore
        self, x_target: Tensor, num_steps: int, task: str, show_progress: bool = False, **kwargs
    ) -> Tensor:
        x_target = x_target.squeeze(1)
        if task == 'continuation':
            y = x_target[..., :self.LEFT_CTX_SIZE]
            z = torch.randn(size=(x_target.size(0), x_target.size(-1) - self.LEFT_CTX_SIZE), device=x_target.device)
            x_noisy = torch.cat([y, z], dim=-1)
        elif task == 'infill':
            y = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], x_target[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
            z = torch.randn((x_target.size(0), x_target.size(-1) - self.LEFT_CTX_SIZE - self.RIGHT_CTX_SIZE), device=x_target.device)
            x_noisy = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], z, x_target[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
        else:
            raise NotImplementedError()
    
        x_noisy = x_noisy.unsqueeze(1)

        b = x_noisy.shape[0]
        sigmas = self.schedule(num_steps + 1, device=x_noisy.device)
        sigmas = repeat(sigmas, "i -> i b", b=b)
        sigmas_batch = extend_dim(sigmas, dim=x_noisy.ndim + 1)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        progress_bar = tqdm(range(num_steps), disable=not show_progress)

        for i in progress_bar:
            v_pred = self.net(x_noisy, sigmas[i], **kwargs)
            x_pred = alphas[i] * x_noisy - betas[i] * v_pred
            noise_pred = betas[i] * x_noisy + alphas[i] * v_pred

            x_noisy = x_noisy.squeeze(1)
            x_noisy.requires_grad_(True)

            with torch.set_grad_enabled(True):
                if task == 'continuation':
                    x_transformed = x_noisy[..., :self.LEFT_CTX_SIZE]
                elif task == 'infill':
                    x_transformed = torch.cat([x_noisy[..., :self.LEFT_CTX_SIZE], x_noisy[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
                else:
                    raise NotImplementedError()

                d_res = self.d(y, x_transformed)
                d_grad = torch.autograd.grad(d_res, x_noisy, grad_outputs=torch.ones_like(d_res))

            d_grad = d_grad[0].unsqueeze(1)
            x_noisy = x_noisy.unsqueeze(1)

            noise_pred -= self.psi * betas[i] * d_grad
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")

        return x_noisy