from pathlib import Path
from typing import List, Dict

import numpy as np


def concatenate_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = list(dicts[0].keys())
    ndim = dicts[0][keys[0]].ndim
    res = {}
    if ndim == 2:
        for key in keys:
            res[key] = np.stack([d[key] for d in dicts], axis=-1)
    else:
        for key in keys:
            res[key] = np.stack([d[key] for d in dicts], axis=3)
    return res

def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )

def get_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()

def get_project_root():
    root = Path(__file__).parent.parent
    return root
