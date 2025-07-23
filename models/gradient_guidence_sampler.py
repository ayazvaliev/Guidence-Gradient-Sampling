import torch
from enum import Enum
from audio_diffusion_pytorch.diffusion import Schedule, LinearSchedule, Sampler, VDiffusion, extend_dim
from torch import Tensor
from math import floor, pi
from typing import Tuple, Callable
import torch.nn as nn
from einops import repeat
from tqdm import tqdm

class Task(Enum):
    CONT: str = 'continuation'
    INFILL: str = 'infill'
    REGEN: str = 'regenerate'
    TRANS: str = 'transition'

class VSamplerWithGradientGuidence(Sampler):
    '''Implementation of waveform DDIM gradient guidence sampling'''

    diffusion_types = [VDiffusion]

    def __init__(self, net: nn.Module, 
                 L_C: float, 
                 R_C: float, 
                 SR: int,  
                 psi = 3 * 1e-3, 
                 k: float = 0.5,
                 f_in: float = 0.5,
                 f_out: float = 0.5,
                 d: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: torch.linalg.vector_norm(y - x, ord=1, dim=-1),
                 schedule: Schedule = LinearSchedule()
                 ):
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
        self.f_in = f_in
        self.f_out = f_out
        self.SR = SR
        self.k = k
        self.d = d
        self.LEFT_CTX_SIZE = floor(self.SR * self.L_C)
        self.RIGHT_CTX_SIZE = floor(self.SR * self.R_C)

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        angle = sigmas * pi / 2
        alpha, beta = torch.cos(angle), torch.sin(angle)
        return alpha, beta

    def _prep_x_noisy(self, task: Task, x_target: torch.Tensor, x_target2: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if task is Task.CONT:
            y = x_target[..., :self.LEFT_CTX_SIZE]
            z = torch.randn(size=tuple(x_target.shape[:-1]) + (x_target.size(-1) - self.LEFT_CTX_SIZE,), device=x_target.device)
            x_noisy = torch.cat([y, z], dim=-1)
        elif task is Task.INFILL:
            y = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], x_target[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
            z = torch.randn(size=tuple(x_target.shape[:-1]) + (x_target.size(-1) - self.LEFT_CTX_SIZE - self.RIGHT_CTX_SIZE,), device=x_target.device)
            x_noisy = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], z, x_target[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
        elif task is Task.REGEN:
            y = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], x_target[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
            z = torch.randn(size=tuple(x_target.shape[:-1]) + (x_target.size(-1) - self.LEFT_CTX_SIZE - self.RIGHT_CTX_SIZE,), device=x_target.device)
            z = self.k * z + (1 - self.k) * x_target[..., self.LEFT_CTX_SIZE:-self.RIGHT_CTX_SIZE]
            x_noisy = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], z, x_target[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
        elif task is Task.TRANS:
            assert x_target2 is not None, 'Second target for Transition task is not passed'
            y = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], x_target2[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
            x_faded = torch.cat(
                [
                    x_target[..., :self.LEFT_CTX_SIZE],
                    self.f_in * x_target[..., self.LEFT_CTX_SIZE: -self.RIGHT_CTX_SIZE] + self.f_out * x_target2[..., self.LEFT_CTX_SIZE: -self.RIGHT_CTX_SIZE],
                    x_target2[..., -self.RIGHT_CTX_SIZE:]
                ],
            dim=-1
            )
            z = torch.randn(size=tuple(x_faded.shape[:-1]) + (x_faded.size(-1) - self.LEFT_CTX_SIZE - self.RIGHT_CTX_SIZE,), device=x_faded.device)
            z = self.k * z + (1 - self.k) * x_target[..., self.LEFT_CTX_SIZE:-self.RIGHT_CTX_SIZE]
            x_noisy = torch.cat([x_faded[..., :self.LEFT_CTX_SIZE], z, x_faded[..., -self.RIGHT_CTX_SIZE:]], dim=-1)

        return x_noisy, y
    
    def _reapply_target_part(self, task: Task, x_noisy: torch.Tensor, x_target: torch.Tensor, x_target2: torch.Tensor | None = None) -> torch.Tensor:
        if task is Task.CONT:
            x_noisy[..., :self.LEFT_CTX_SIZE] = x_target[..., :self.LEFT_CTX_SIZE]
        elif task is Task.INFILL or task is Task.REGEN:
            x_noisy = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], x_noisy[..., self.LEFT_CTX_SIZE:-self.RIGHT_CTX_SIZE], x_noisy[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
        elif task is Task.TRANS:
            x_noisy = torch.cat([x_target[..., :self.LEFT_CTX_SIZE], x_noisy[..., self.LEFT_CTX_SIZE:-self.RIGHT_CTX_SIZE], x_target2[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
        return x_noisy

    def _calculate_d(self, x_noisy: torch.Tensor, y: torch.Tensor, task: Task) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            if task is Task.CONT:
                x_transformed = x_noisy[..., :self.LEFT_CTX_SIZE]
            elif task is Task.INFILL or task is Task.REGEN or task is Task.TRANS:
                x_transformed =  x_transformed = torch.cat([x_noisy[..., :self.LEFT_CTX_SIZE], x_noisy[..., -self.RIGHT_CTX_SIZE:]], dim=-1)
            d_res = self.d(y, x_transformed)
        return d_res

    @torch.no_grad()
    def forward(  # type: ignore
        self, 
        x_target: Tensor, 
        num_steps: int, 
        task: str, 
        show_progress: bool = False, 
        **kwargs
    ) -> Tensor:
        try:
            task = Task(task)
        except ValueError:
            raise NotImplementedError('Task passed to sampler is not supported')

        x_target2 = kwargs.get('x_target2', None)
        x_noisy, y = self._prep_x_noisy(task, x_target, x_target2)

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

            x_noisy.requires_grad_(True)
            d_res = self._calculate_d(x_noisy, y, task)
            d_grad = torch.autograd.grad(d_res, x_noisy)[0]

            # d_grad = d_grad.unsqueeze(1)
            # x_noisy = x_noisy.unsqueeze(1)

            noise_pred -= self.psi * betas[i] * d_grad
            x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
            progress_bar.set_description(f"Sampling (noise={sigmas[i+1,0]:.2f})")
        
        x_noisy = self._reapply_target_part(x_noisy, x_target, x_target2)
        return x_noisy