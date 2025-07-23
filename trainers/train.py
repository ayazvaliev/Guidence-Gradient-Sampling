from models.gradient_guidence_sampler import VSamplerWithGradientGuidence
from trainers.evaluator import Evaluator
from data.dataset import FMADataset
from utils.load_dataset import load_dataset, get_paths
from utils.misc import count_parameters

from audio_diffusion_pytorch import UNetV0, VDiffusion, DiffusionModel
from torch.utils.data import DataLoader
from utils.device import DEVICE

import torch
import yaml
from torch import optim
import numpy as np
from tqdm import tqdm
import os


with open('configs/train_configs.yaml', 'r') as yaml_f:
    train_params = yaml.safe_load(yaml_f)

with open('configs/unet_configs.yaml') as yaml_f:
    unet_params = yaml.safe_load(yaml_f)
 

def train(dataset_path:str, download_dataset:bool = True, do_eval:bool = True, logger: bool=True):

    ratio = train_params['TRAIN_TEST_SPLIT_RATIO'] if do_eval else 1.0
    if download_dataset:
        train_paths, test_paths = get_paths(load_dataset(dataset_path), ratio)
    else:
        train_paths, test_paths = get_paths(dataset_path, ratio)
    ratio = train_params['TRAIN_TEST_SPLIT_RATIO'] if do_eval else 1.0
    sr = train_params['SR']
    crop_s = train_params['CROP_S']
    if do_eval:
        evaluator = Evaluator(sr=sr, model=train_params['EMBEDDING_EXTRACTOR_MODEL'], device=DEVICE) if do_eval else None
    else:
        evaluator = None

    dataset = FMADataset(train_paths, sr=sr, crop_s=crop_s)
    test_dataset = FMADataset(test_paths, sr=sr, crop_s=crop_s)

    loader  = DataLoader(dataset, train_params['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_dataset, train_params['BATCH_SIZE'], shuffle=False, drop_last=False)

    epochs = train_params['EPOCHS']
    TOTAL_STEPS = epochs * len(loader)

    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=unet_params['IN_CHANNELS'],
        channels=unet_params['CHANNELS'],
        factors=unet_params['FACTORS'], 
        items=unet_params['ITEMS'],
        attentions=unet_params['ATTENTIONS'],
        attention_heads=unet_params['ATTENTION_HEADS'],
        attention_features=unet_params['ATTENTION_FEATURES'],
        diffusion_t=VDiffusion, 
        sampler_t=VSamplerWithGradientGuidence, 
        sampler_L_C=train_params['L_C'],
        sampler_R_C=train_params['R_C'],
        sampler_SR=train_params['SR'],
    )
    model.to(DEVICE)

    if logger:
        print(f'len(train_dataset)={len(dataset)}, len(eval_dataset)={len(test_dataset)}')
        print(f'Total Unet parameters: {count_parameters(model.net)}')

    optimizer = optim.AdamW(model.parameters(), lr=float(train_params['BASE_LR']), betas=(0.9, 0.999), weight_decay=0.0)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1,
        total_iters=train_params['WARMUP_STEPS']
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        TOTAL_STEPS - train_params['WARMUP_STEPS']
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[train_params['WARMUP_STEPS']]
    )

    fads_history_continuation = []
    fads_history_infill = []
    fads_history_regen = []
    fads_history_transition = []
    mrs_history_continuation = []
    mrs_history_infill = []
    mrs_history_regen = []
    mrs_history_transition = []
    eval_epochs_history = []

    accum_iter = train_params['ACCUM_GRADS'] // train_params['BATCH_SIZE']
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        train_iterator = tqdm(loader, f'Epoch {epoch}/{epochs}') if logger else loader
        
        for batch_idx, batch in enumerate(train_iterator):
            batch = batch.to(DEVICE)
            loss = model(batch)
            loss = loss / accum_iter
            loss.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if logger:
                    losses.append(loss.item())
            
        if logger:
            print(f'Epoch loss: {np.mean(np.asarray(losses))}')

        if epoch % train_params['CKPT_INTERVAL'] == 0:
            if train_params['CKPT_DIR'] is not None:
                os.makedirs(train_params['CKPT_DIR'], exist_ok=True)
                torch.save(
                {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
                f"{train_params['CKPT_DIR']}/ckpt_e{epoch:04d}.pt",
            )
            if do_eval:
                assert evaluator is not None

                fads_continuation = []
                fads_infill = []
                fads_regen = []
                fads_transition = []
                mrs_continuation = []
                mrs_infill = []
                mrs_regen = []
                mrs_transition = []

                model.eval()

                for samples in test_loader:
                    samples = samples.to(DEVICE)

                    samples_continuation = model.sample(
                        task='continuation', 
                        x_target=samples, 
                        num_steps=train_params['SAMPLING_STEPS_EVAL']
                        )

                    samples_infill = model.sample(
                        task='infill', 
                        x_target=samples, 
                        num_steps=train_params['SAMPLING_STEPS_EVAL'])

                    samples_regen = model.sample(
                        task='regenerate', 
                        x_target=samples, 
                        num_steps=train_params['SAMPLING_STEPS_EVAL'])
                    
                    samples_mask = [i % 2 for i in range(samples.size(0) - (samples.size(0) % 2))]
                    if samples.size(0) % 2 != 0:
                        samples_mask.append(False)
                    samples_mask = torch.tensor(samples_mask, dtype=torch.bool, device=DEVICE)
                    samples_1 = samples[samples_mask]
                    samples_2 = samples[~samples_mask]
                    samples_transition = model.sample(
                        task='transition', 
                        x_target=samples_1, 
                        num_steps=train_params['SAMPLING_STEPS_EVAL'], 
                        x_target2=samples_2)


                    fad_continuation, mr_continuation = evaluator.calculate_fad_and_mr(samples, samples_continuation)
                    fad_infill, mr_infill = evaluator.calculate_fad_and_mr(samples, samples_infill)
                    fad_regen, mr_regen = evaluator.calculate_fad_and_mr(samples, samples_regen)
                    fad_transition_1, mr_transition_1 = evaluator.calculate_fad(samples_transition, samples_1)
                    fad_transition_2, mr_transition_2 = evaluator.calculate_fad(samples_transition, samples_2)
                    fad_transition = (fad_transition_1 + fad_transition_2) / 2
                    mr_transition = (mr_transition_1 + mr_transition_2) / 2

                    fads_continuation.append(fad_continuation)
                    fads_infill.append(fad_infill)
                    fads_transition.append(fad_transition)
                    fads_regen.append(fad_regen)
                    mrs_continuation.append(mr_continuation)
                    mrs_infill.append(mr_infill)
                    mrs_transition.append(mr_transition)
                    mrs_regen.append(mr_regen)
            
                fads_history_continuation.append(np.mean(np.asarray(fads_continuation)))
                fads_history_infill.append(np.mean(np.asarray(fads_infill)))
                fads_history_transition.append(np.mean(np.asarray(fads_transition)))
                fads_history_regen.append(np.mean(np.asarray(fads_regen)))
                mrs_history_continuation.append(np.mean(np.asarray(mrs_continuation)))
                mrs_history_infill.append(np.mean(np.asarray(mrs_infill)))
                mrs_history_transition.append(np.mean(np.asarray(mrs_transition)))
                mrs_history_regen.append(np.mean(np.asarray(mrs_regen)))
                eval_epochs_history.append(epoch)

                if logger:
                    print(f'''Epoch={epoch}
                    FAD_continuation={fads_history_continuation[-1]},
                    FAD_infill={fads_history_infill[-1]},
                    FAD_regen={fads_history_regen[-1]},
                    FAD_transition={fads_history_transition[-1]}
                    MR_continuation={mrs_history_continuation[-1]},
                    MR_infill={mrs_history_infill[-1]},
                    MR_regen={mrs_history_regen[-1]},
                    MR_transition={mrs_history_transition[-1]}''')
    return model
