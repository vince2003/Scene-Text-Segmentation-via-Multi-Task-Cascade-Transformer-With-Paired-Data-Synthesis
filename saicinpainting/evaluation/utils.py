from enum import Enum

import yaml
from easydict import EasyDict as edict
import torch.nn as nn
import torch
import pdb

def load_yaml(path):
    with open(path, 'r') as f:
        return edict(yaml.safe_load(f))


def move_to_device(obj, device):
    #pdb.set_trace()
    
    # if obj.pop('name', None)!=None:
    #     del obj['name']
    
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        del obj['name']
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')


class SmallMode(Enum):
    DROP = "drop"
    UPSCALE = "upscale"
