import numpy as np

import os
import tqdm
import argparse


__all__ = ["embd2hashArgs", "embd2hash"]


def embd2hashArgs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-E", type=str, required=True, dest="embd_load_fold",
    )
    parser.add_argument(
        "-H", type=str, required=True, dest="hash_save_fold",
    )


def embd2hash(
    embd_load_fold: str, hash_save_fold: str, *vargs, **kwargs
) -> None:
    for chromosome in tqdm.tqdm(
        [str(i) for i in range(1, 23)] + ["X"],     # BAM naming convention
        unit="chr", desc=hash_save_fold, smoothing=0.0, dynamic_ncols=True,
    ):
        # read embd of given sample id and chromosome
        embd = np.load(os.path.join(embd_load_fold, f"{chromosome}.npy"))
        embd = embd[embd[:, 768].argsort()]
        # bucket and bucket2pos
        bucket, bucket2pos = np.unique(embd[:,  768]//1000, return_index=True)
        bucket2pos = np.concatenate((bucket2pos, [len(embd)]))
        bucket = bucket.astype(int).tolist()
        bucket2pos = bucket2pos.astype(int).tolist()
        # store file in bucket
        for b  in tqdm.tqdm(
            range(len(bucket)), leave=False,
            unit="bucket", desc=chromosome, smoothing=0.0, dynamic_ncols=True,
        ):
            embd_bucket = embd[bucket2pos[b]:bucket2pos[b+1]]
            bucket_str = f"{bucket[b]:06d}"
            fold_name = bucket_str[:3]
            file_name = bucket_str[3:] + ".npy"
            os.makedirs(
                os.path.join(hash_save_fold, chromosome, fold_name),
                exist_ok=True
            )
            np.save(
                os.path.join(hash_save_fold, chromosome, fold_name, file_name), 
                embd_bucket
            )
