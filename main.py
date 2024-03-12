import torch
import torch.cuda
import torch.backends.cudnn
import transformers

import numpy as np

import os
import random
import warnings

import src


__all__ = []


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Unable to import Triton*")


def main():
    setSeed(42)
    config = src.Config()
    src.Trainer(**config.runner,
        trainset=src.DNADataset(**config.trainset),
        model=src.DNABERT2FC(**config.model),
    ).fit()


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    np.random.seed(seed)
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  # python hash seed


if __name__ == "__main__": main()
