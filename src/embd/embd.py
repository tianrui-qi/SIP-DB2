import torch
import torch.cuda
import torch.backends.cudnn
import transformers

import numpy as np

import os
import random
import argparse
import warnings

from .cfg import Config
from .data import FinetuneDataset
from .model import FinetuneModel
from .runner import Trainer


__all__ = ["embdArgs", "embd"]


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Unable to import Triton*")


def embdArgs(parser: argparse.ArgumentParser) -> None:
    pass


def embd(*vargs, **kwargs):
    # set seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)            # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)    # python hash seed
    # finetune
    config = Config()
    Trainer(**config.runner,
        trainset=FinetuneDataset(**config.trainset),
        model=FinetuneModel(**config.model),
    ).fit()
