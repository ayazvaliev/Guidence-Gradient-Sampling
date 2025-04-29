from pipelines.train import train
from config.model_params import MODEL_DIR

import argparse
import torch

parser = argparse.ArgumentParser('Trains DDIM Waveform from scratch (all changable parameters are stored in config/model_params.py)')
parser.add_argument('--do_eval', help='Evaluate metrics on test dataset during training', action='store_true')
parser.add_argument('--logging', help='Enables logging during training', action='store_true')
parser.add_argument('--save_model', help='Saves model\'s state dict after training', action='store_true')
parser.add_argument('--load_dataset', help='Loads dataset into passed destination path', action='store_true')
parser.add_argument('-d', '--destination', type=str, help='Path to directory with raw .mp3 or .wav audio files (algorithm will crawl through directory recursively)')

if __name__ == '__main__':
    args = parser.parse_args()
    model = train(
        destination_path=args.destination,
        load_dataset=args.load_dataset,
        do_eval=args.do_eval,
        logger=args.logging
    )
    if args.save_model:
        torch.save(model.state_dict(), f"{MODEL_DIR}/model_final.pt")