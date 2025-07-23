import torch
from models.gradient_guidence_sampler import VSamplerWithGradientGuidence
from trainers.evaluator import Evaluator
from utils.load_dataset import get_paths
from data.dataset import FMADataset
from torch.utils.data import DataLoader
from audio_diffusion_pytorch import VDiffusion, UNetV0, DiffusionModel
import soundfile as sf
from utils.device import DEVICE
from pathlib import Path
import numpy as np
import yaml


with open('configs/train_configs.yaml', 'r') as yaml_f:
    train_params = yaml.safe_load(yaml_f)


with open('configs/unet_configs.yaml') as yaml_f:
    unet_params = yaml.safe_load(yaml_f)


def sample_from_pretrained(model_path: str, 
                           num_steps: int, 
                           task: str, 
                           dataset_path: str, 
                           repeats: int = 1, 
                           save_dir: str | None = None, 
                           save_format: str = 'wav',
                           show_progress: bool = False):
    paths, _ = get_paths(dataset_path)

    evaluator = Evaluator(sr=train_params['SR'], model=train_params['EMBEDDING_EXTRACTOR_MODEL'], device=DEVICE)
    test_dataset = FMADataset(paths, sr=train_params['SR'], crop_s=train_params['CROP_S'], return_name=True)
    assert task != 'transition' or len(test_dataset) == 2, 'Transition task requires only two audiofiles in the dataset'
    test_loader = DataLoader(test_dataset, train_params['BATCH_SIZE'], shuffle=False, drop_last=False)

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
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model'])
    fads_per_iter = []
    mrs_per_iter = []

    for repeat in range(repeats):
        fads = []
        mrs = []
        if task == 'transition':
            test_loader_iter = iter(test_loader)
            samples, samples_name = next(test_loader_iter)
            if samples.size(0) < 2:
                sample_1 = samples
                sample_2, sample_name_2 = next(test_loader_iter)
                sample_name_1 = samples_name[0]
                sample_name_2 = sample_name_2[0]
            else:
                sample_1, sample_2 = samples[0].unsqueeze(0), samples[-1].unsqueeze(0)
                sample_name_1, sample_name_2 = samples_name

            sample_gen = model.sample(task, 
                                      sample_1, 
                                      num_steps=num_steps, 
                                      x_target2=sample_2,
                                      show_progress=show_progress)
            
            
            filename = Path(sample_name_1 + '+' + sample_name_2 + '.' + save_format)
            sf.write(Path(save_dir) / filename, sample_gen.cpu().numpy(), train_params['SR'])
            fad, mr = evaluator.calculate_fad_and_mr(sample_1, samples_gen)
            fad2, mr2 = evaluator.calculate_fad_and_mr(sample_2, sample_gen)
            
            fads.append((fad + fad2) / 2)
            mrs.append((mr + mr2) / 2)
        else:
            for samples, samples_name in test_loader:
                samples = samples.to(DEVICE)
                samples_gen = model.sample(task, 
                                           samples,
                                           num_steps=num_steps,
                                           show_progress=show_progress)
        
                if save_dir is not None and repeat == repeats - 1:
                    for sample_gen, name in zip(samples_gen, samples_name):
                        filename = Path(name + '.' + save_format)
                        sf.write(Path(save_dir) / filename, sample_gen.cpu().numpy(), train_params['SR'])

                fad, mr = evaluator.calculate_fad_and_mr(samples, samples_gen)
                fads.append(fad)
                mrs.append(mr)

        fads_per_iter.append(np.mean(np.asarray(fads)))
        mrs_per_iter.append(np.mean(np.asarray(mr)))

    return len(test_dataset), np.asarray(fads_per_iter), np.asarray(mrs_per_iter)