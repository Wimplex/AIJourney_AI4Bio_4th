import os
import json
import shutil
import random
import pickle
import pathlib
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch


def read_data(json_path: str):
    with open(json_path, 'rb') as fp:
       data = json.load(fp)
    df = pd.DataFrame(json.loads(data)).T
    return df


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_top_n_indices(arr: np.ndarray, n: int = 10) -> np.ndarray:
    ids = np.argsort(arr)
    return ids[-n:]


def min_max_scale_2d(matrix: np.ndarray, min_: float = None, max_: float = None) -> np.ndarray:
    if min_ is None or max_ is None:
        min_, max_ = np.min(matrix), np.max(matrix)
    return (matrix - min_) / (max_ - min_), min_, max_


def save_model(model_dict: Dict[str, Any], save_path: str) -> None:
    with open(save_path, "wb") as file:
        pickle.dump(model_dict, file)


def load_model(load_path: str) -> None:
    with open(load_path, "rb") as file:
        model_dict = pickle.load(file)
    return model_dict


def check_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clear_dir(dir_path: pathlib.Path):
    files = dir_path.glob("*")
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)


def softmax(a: np.ndarray):
    return np.exp(a) / (np.sum(np.exp(a)) + 1e-10)


def invert_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}