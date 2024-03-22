import cuml
import numpy as np

import gc
import os
import tqdm
import joblib
import argparse


__all__ = ["umapArgs", "umap"]


def umapArgs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-E", type=str, required=True, dest="embd_load_fold",
        help="Fold to load the embedding. Assume the sturcture is " + 
        "`embd_load_fold/$id/$chr.npy` where [0:768] of each $chr.npy is the " +
        "embedding."
    )
    parser.add_argument(
        "-C", type=str, required=True, dest="ckpt_save_fold",
        help="Fold to save fitted UMAP. For each chromosome, the UMAP will be " + 
        "calculated on all samples' embedding and saved as " +
        "`ckpt_save_fold/$chr.pkl` using joblib."
    )


def umap(embd_load_fold: str, ckpt_save_fold: str, *vargs, **kwargs) -> None:
    for chr in tqdm.tqdm(
        [str(i) for i in range(1, 23)] + ["X"], 
        desc=ckpt_save_fold, unit="chr", smoothing=0, dynamic_ncols=True,
    ):
        # check if umap of current chromosome already exists
        if os.path.exists(os.path.join(ckpt_save_fold, f"{chr}.pkl")):
            continue
        # get pretrain embedding of current chr of all sample
        pretrain = None
        for run in tqdm.tqdm(
            os.listdir(embd_load_fold), 
            desc=chr, unit="run", smoothing=0, dynamic_ncols=True, leave=False, 
        ):
            pretrain_run = np.load(
                os.path.join(embd_load_fold, run, f"{chr}.npy")
            )[:, 0:768]
            if pretrain is None:
                pretrain = pretrain_run
            else:
                pretrain = np.concatenate((pretrain, pretrain_run))
        # calculate the umap and save the umap
        reducer = cuml.UMAP(n_components=2)
        reducer.fit(pretrain)
        if not os.path.exists(ckpt_save_fold):
            os.makedirs(ckpt_save_fold)
        joblib.dump(reducer, os.path.join(ckpt_save_fold, f"{chr}.pkl"))
        # release memory
        # refer to https://github.com/rapidsai/cuml/issues/5666
        gc.collect()
