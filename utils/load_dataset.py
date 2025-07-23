import os
import glob
import yaml
from random import shuffle
from pathlib import Path
import requests
from yarl import URL
import zipfile
import tempfile


def load_dataset(dest_path: str, config_path: Path = Path('configs/dataset_configs.yaml')) -> str:
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    try:
        links = dataset_config['DATASET_URLS']
        for link in links:
            link = URL(link)
            response = requests.get(link)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=link.name.split('.')[-1], delete=True) as zip_file:
                with open(zip_file, 'wb') as f:
                    f.write(response.content)
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    os.makedirs(dest_path, exist_ok=True)
                    zip_ref.extractall(path=dest_path)
        return dest_path
    except KeyError:
        raise KeyError('Unable to download dataset: Config file missing DATASET_URLS keyword')

def get_paths(dataset_path: str, split_ratio: float = 1.0) -> tuple[list[str], list[str]]:
    '''
    dataset_path: path to audio dataset
    returns: list of filtered paths to audio files
    '''

    exts = ('*.wav', '*.mp3')
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(dataset_path, "**", ext), recursive=True)
    paths = [p for p in paths if os.path.getsize(p) > 100 * 1024]
    shuffle(paths)
    anchor = int(len(paths) * split_ratio)
    train_paths, test_paths = paths[:anchor], paths[anchor:]

    return train_paths, test_paths