from core.gradient_guidence_sampler import VSamplerWithGradientGuidence
from pipelines.evaluator import Evaluator
from core.fmadataset import FMADataset, get_paths, load_dataset
from config.model_params import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    CKPT_INT,
    WARMUP_STEPS,
    SR,
    CROP_S,
    L_C,
    R_C,
    ACCUM_ITER,
    DEVICE,
    IN_CHANNELS,
    CHANNELS,
    FACTORS,
    ITEMS,
    ATTENTIONS,
    ATTENTION_HEADS,
    ATTENTION_FEATURES,
    TRAIN_TEST_SPLIT_RATIO,
    EMBEDDING_EXTRACTOR_MODEL,
    CHECKPOINTS_OUT_DIR,
    NUM_STEPS_FOR_EVALUATION
)
from core.utils import count_parameters

from audio_diffusion_pytorch import VDiffusion, UNetV0, DiffusionModel
from torch.utils.data import DataLoader

import torch
from torch import optim
import math
import os
import numpy as np
from tqdm import tqdm

device = DEVICE

class LRScheduler:
    def __init__(self, total_steps, warmup_steps):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, self.total_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

def train(destination_path, load_dataset=True, do_eval=True, logger=True):
    if load_dataset:
        paths = get_paths(load_dataset(destination_path))
    else:
        paths = get_paths(destination_path)

    ratio = TRAIN_TEST_SPLIT_RATIO if do_eval else 1.0
    evaluator = Evaluator(sr=SR, model=EMBEDDING_EXTRACTOR_MODEL, device=device) if do_eval else None


    dataset = FMADataset(paths, 'train', sr=SR, crop_s=CROP_S, ratio=ratio)
    test_dataset = FMADataset(paths, 'test', sr=SR, crop_s=CROP_S, ratio=ratio)

    loader  = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=False)

    TOTAL_STEPS = EPOCHS * len(loader)

    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=IN_CHANNELS,
        channels=CHANNELS,
        factors=FACTORS, 
        items=ITEMS, 
        attentions=ATTENTIONS, 
        attention_heads=ATTENTION_HEADS, 
        attention_features=ATTENTION_FEATURES, 
        diffusion_t=VDiffusion, 
        sampler_t=VSamplerWithGradientGuidence, 
        sampler_L_C=L_C,
        sampler_R_C=R_C,
        sampler_SR=SR,
    )
    model.to(DEVICE)

    if logger:
        print(f'len(train_dataset)={len(dataset)}, len(eval_dataset)={len(test_dataset)}')
        print(f'Total Unet parameters: {count_parameters(model.net)}')


    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, LRScheduler(TOTAL_STEPS, WARMUP_STEPS))

    fads_history_continuation = []
    fads_history_infill = []
    mrs_history_continuation = []
    mrs_history_infill = []
    eval_epochs_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        train_iterator = tqdm(loader, f'Epoch {epoch}/{EPOCHS}') if logger else loader
        
        for batch_idx, batch in enumerate(train_iterator):
            batch = batch.to(device)
            loss = model(batch)
            loss = loss / ACCUM_ITER
            loss.backward()

            if ((batch_idx + 1) % ACCUM_ITER == 0) or (batch_idx + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if logger:
                    losses.append(loss.item())
            
        if logger:
            print(f'Epoch loss: {np.mean(np.asarray(losses))}')

        if epoch % CKPT_INT == 0:
            if CHECKPOINTS_OUT_DIR is not None:
                torch.save(
                {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
                f"{CHECKPOINTS_OUT_DIR}/ckpt_e{epoch:04d}.pt",
            )
            if do_eval:
                assert evaluator is not None

                fads_continuation = []
                fads_infill = []
                mrs_continuation = []
                mrs_infill = []

                model.eval()

                with torch.no_grad():
                    for samples in test_loader:
                        samples = samples.to(device)

                        samples_continuation = model.sample(samples, num_steps=NUM_STEPS_FOR_EVALUATION, task='continuation')
                        samples_infill = model.sample(samples, num_steps=NUM_STEPS_FOR_EVALUATION, task='infill')

                        fad_continuation, mr_continuation = evaluator.calculate_fad_and_mr(samples, samples_continuation)
                        fad_infill, mr_infill = evaluator.calculate_fad_and_mr(samples, samples_infill)

                        fads_continuation.append(fad_continuation)
                        fads_infill.append(fad_infill)
                        mrs_continuation.append(mr_continuation)
                        mrs_infill.append(mr_infill)
            
                    fads_history_continuation.append(np.mean(np.asarray(fads_continuation)))
                    fads_history_infill.append(np.mean(np.asarray(fads_infill)))
                    mrs_history_continuation.append(np.mean(np.asarray(mrs_continuation)))
                    mrs_history_infill.append(np.mean(np.asarray(mrs_infill)))
                    eval_epochs_history.append(epoch)

                    if logger:
                        print(f'''Epoch={epoch}
                        FAD_continuation={fads_history_continuation[-1]},
                        FAD_infill={fads_history_infill[-1]},
                        MR_continuation={mrs_history_continuation[-1]},
                        MR_infill={mrs_history_infill[-1]}''')
    return model
