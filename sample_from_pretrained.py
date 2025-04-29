import argparse
from core.sampling import sample_from_pretrained

parser = argparse.ArgumentParser(description='Samples audio waveforms with defined task and model weights, evaluates and prints out metrics')
parser.add_argument('--model_path', type=str, help='Path to DiffusionModel weights dictionary')
parser.add_argument('--num_steps', type=int, help='Number of denoising steps in sampling procedure')
parser.add_argument('--dataset_path', type=str, help='Path to directory with raw .mp3 or .wav audio files')
parser.add_argument('--task', type=str, help='Defines task for sampler')

if __name__ == '__main__':
    args = parser.parse_args()
    len_samples, fad, mr = sample_from_pretrained(
        args.model_path,
        args.num_steps,
        args.task,
        args.dataset_path
    )
    print(f'''Total samples = {len_samples},
          FAD = {fad},
          MR = {mr}''')