import torch
from core.gradient_guidence_sampler import VSamplerWithGradientGuidence
from pipelines.evaluator import Evaluator
from core.fmadataset import FMADataset, get_paths
from torch.utils.data import DataLoader
from audio_diffusion_pytorch import VDiffusion, UNetV0, DiffusionModel
import numpy as np
from config.model_params import (
    SR,
    CROP_S,
    L_C,
    R_C,
    DEVICE,
    IN_CHANNELS,
    CHANNELS,
    FACTORS,
    ITEMS,
    ATTENTIONS,
    ATTENTION_HEADS,
    ATTENTION_FEATURES,
    EMBEDDING_EXTRACTOR_MODEL,
    BATCH_SIZE,
)

implemented_tasks = ['continuation', 'infill']

def sample_from_pretrained(model_path: str, num_steps: int, task: str, dataset_path: str):
    if task not in implemented_tasks:
        raise NotImplementedError('Selected task is not implemented in sampler')

    paths = get_paths(dataset_path)

    evaluator = Evaluator(sr=SR, model=EMBEDDING_EXTRACTOR_MODEL, device=DEVICE)

    test_dataset = FMADataset(paths, dataset_type='train', ratio=1, sr=SR, crop_s=CROP_S)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=False)

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
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model'])
    
    fads = []
    mrs = []
    with torch.no_grad():
        for samples in test_loader:
            samples = samples.to(DEVICE)
            samples_gen = model.sample(samples, num_steps=num_steps, task=task)
            
            fad, mr = evaluator.calculate_fad_and_mr(samples, samples_gen)
            fads.append(fad)
            mrs.append(mr)
    
    return len(test_dataset), np.mean(np.asarray(fads)), np.mean(np.asarray(mr))