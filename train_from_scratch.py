from trainers.train import train
import argparse
import torch

parser = argparse.ArgumentParser('Trains Waveform DDIM from scratch (all changeable parameters are stored in config/model_params.py)')
parser.add_argument('--do_eval', help='Evaluate metrics on test dataset during training', action='store_true')
parser.add_argument('--do_logging', help='Enables logging during training', action='store_true')
parser.add_argument('--save_model_dir', type=str, default='',help='Saves model\'s state dict after training')
parser.add_argument('--download_dataset', help='Loads dataset into passed destination path', action='store_true')
parser.add_argument('-d', '--dataset', type=str, default='', help='Path to directory with raw .mp3 or .wav audio files (algorithm will crawl through directory recursively)')


if __name__ == '__main__':
    args = parser.parse_args()
    model = train(
        dataset_path=args.dataset,
        download_dataset=args.download_dataset,
        do_eval=args.do_eval,
        logger=args.do_logging
    )
    if args.save_model_dir != '':
        torch.save(model.state_dict(), f"{args.save_model_dir}/model_final.pt")