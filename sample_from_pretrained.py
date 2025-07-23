import argparse
import numpy as np
from trainers.sample import sample_from_pretrained

parser = argparse.ArgumentParser(description='Samples audio waveforms with defined task and model weights, evaluates and prints out metrics')
parser.add_argument('--model_path', type=str, help='Path to DiffusionModel weights dictionary')
parser.add_argument('--num_steps', type=int, help='Number of denoising steps in sampling procedure')
parser.add_argument('--dataset_path', type=str, help='Path to directory with raw .mp3 or .wav audio files')
parser.add_argument('--save_dir', type=str, default='', help='Path to directory with sampled results (optional)')
parser.add_argument('--save_format', type=str, default='wav', help='Audiofile format for sampled results')
parser.add_argument('--task', type=str, help='Defines task for sampler. Supported tasks: continuation, infill, renegerate, transition')
parser.add_argument('--repeats', type=int, default=1, help='Defines number of iterations of evaluation for closer-to-reality metrics values')

if __name__ == '__main__':
    args = parser.parse_args()
    len_samples, fad_per_iter, mr_per_iter = sample_from_pretrained(
        args.model_path,
        args.num_steps,
        args.task,
        args.dataset_path,
        args.repeats,
        args.save_dir if args.save_dir != '' else None,
        args.save_format
    )

    print(f'Total sampels = {len_samples}')
    for i in range(1, args.repeats + 1):
        print(f'''Iteration #{i}:
              FAD={fad_per_iter[i]},
              MR={mr_per_iter[i]}''')
        print('\n')
    print(f'''Averaged FAD: {np.mean(fad_per_iter)},
          Averaged MR: {np.mean(mr_per_iter)}''')