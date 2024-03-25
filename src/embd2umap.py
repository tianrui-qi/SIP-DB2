import numpy as np

import os
import gc
import tqdm
import joblib
import argparse


__all__ = ["embd2umapArgs", "embd2umap"]


def embd2umapArgs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-E", type=str, required=True, dest="embd_load_fold",
        help="Fold to load the embedding of one sample. Assume the sturcture " +
        "is `embd_load_fold/$chr.npy` where [0:768] of each $chr.npy is the " +
        "embedding."
    )
    parser.add_argument(
        "-C", type=str, required=True, dest="ckpt_load_fold",
        help="Fold to load UMAP checkpoint for transform. Assume the " +
        "structure is `ckpt_load_fold/$chr.pkl` where $chr.pkl is the " +
        "UMAP checkpoint for chromosome $chr."
    )
    parser.add_argument(
        "-U", type=str, required=True, dest="umap_save_fold",
        help="Fold to save UMAP transform result. For each chromosome, we " + 
        "use a seperate UMAP checkpoint to reduce the embedding dimension " +
        "from 768 to 2 and save the result as `umap_save_fold/$chr.npy`, a " + 
        "N by 10 numpy array where N for number of reads. For each read, " + 
        "`[0:2]` is the UMAP1 and UMAP2, `[2]`is the pos, and `[3:10]` is " + 
        "num of variants cover by each read with different p-value " + 
        "threshold. Note that `[2:10]` directly copy from `[768:776]` of " + 
        "`-E EMBD_LOAD_FOLD`. If UMAP result `$chr.npy` already exists or " + 
        "UMAP checkpoint `$chr.pkl` not exists, skip the chromosome."
    )
    parser.add_argument(
        "-p", type=float, required=False, dest="pval_thresh", default=1e-04,
        help="P-value threshold for filtering reads, i.e., only keep reads " + 
        "that cover at least one variant with p-value <= pval_thresh. " + 
        "Default: 1e-04."
    )
    parser.add_argument(
        "-b", type=int, required=False, dest="batch_size", default=100,
        help="Batch size for number of reads input to UMAP to perform " + 
        "transform at same time. Default: 100."
    )


def embd2umap(
    embd_load_fold: str, ckpt_load_fold: str, umap_save_fold: str,
    pval_thresh: float = 1e-04, batch_size: int = 100, *vargs, **kwargs,
) -> None:
    for chr in tqdm.tqdm(
        [str(i) for i in range(1, 23)] + ["X"],     # BAM naming convention
        unit="chr", desc=umap_save_fold, smoothing=0.0, dynamic_ncols=True,
    ):
        embd_load_path = os.path.join(embd_load_fold, f"{chr}.npy")
        ckpt_load_path = os.path.join(ckpt_load_fold, f"{chr}.pkl")
        umap_save_path = os.path.join(umap_save_fold, f"{chr}.npy")

        # load the embedding and filter by pval_thresh
        embd = np.load(embd_load_path)
        embd = embd[embd[:, 769-int(np.log10(pval_thresh))]>=1, :]

        # check if umap already calculated
        if os.path.exists(umap_save_path): 
            umap = np.load(umap_save_path)
            if len(umap) >= len(embd):
                tqdm.tqdm.write(f"{umap_save_path} already calculated, skip.")
                continue
        # check if ckpt exists
        if not os.path.exists(ckpt_load_path): 
            tqdm.tqdm.write(f"{ckpt_load_path} not exists, skip.")
            continue

        # calculate umap
        reducer = joblib.load(ckpt_load_path)
        umap = None
        for i in tqdm.tqdm(
            range(int(np.ceil(len(embd)/batch_size))), desc=chr, leave=False,
            unit="batch", smoothing=0.0, dynamic_ncols=True,
        ):
            umap_batch = reducer.transform(
                embd[i*batch_size:(i+1)*batch_size, 0:768]
            )
            umap = np.concatenate(
                [umap, umap_batch], axis=0
            ) if umap is not None else umap_batch
        umap = np.concatenate((umap, embd[:, 768:]), axis=1)

        # save
        if not os.path.exists(umap_save_fold): os.makedirs(umap_save_fold)
        np.save(umap_save_path, umap)

        # release memory
        # refer to https://github.com/rapidsai/cuml/issues/5666
        gc.collect()
