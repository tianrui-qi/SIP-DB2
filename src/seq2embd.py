import torch
import transformers

import numpy as np
import pandas as pd

import os
import tqdm
import warnings

from src.embd.model import PretrainModel, FinetuneModel


__all__ = ["seq2embd"]


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton*")


def seq2embd(
    hdf_load_path: str, embd_save_fold: str, 
    ckpt_load_path: str = None,
    pval_thresh: float = 0, batch_size: int = 100,
    *vargs, **kwargs
) -> None:
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    if ckpt_load_path is None:
        model = PretrainModel()
    else:
        ckpt = torch.load(ckpt_load_path)
        model = FinetuneModel(
            feats_token=ckpt["feats_token"], feats_coord=ckpt["feats_coord"], 
            feats_final=ckpt["feats_final"]
        )
        model.fc_token.load_state_dict(ckpt['fc_token'])
        model.fc_coord.load_state_dict(ckpt['fc_coord'])
    model.eval().to(device)

    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for c in tqdm.tqdm(
        [str(i) for i in range(1, 23)] + ["X"],     # BAM naming convention
        unit="chr", desc=embd_save_fold, smoothing=0.0, dynamic_ncols=True,
    ):
        # get the hdf that store sequence and p-value for filtering
        hdf = pd.read_hdf(hdf_load_path, key=f"/chr{c}", mode="r")
        if pval_thresh != 0: hdf = hdf[hdf[f"{pval_thresh:.0e}"]>=1]

        embd = None
        for i in tqdm.tqdm(
            range(int(np.ceil(len(hdf)/batch_size))), desc=c, leave=False,
            unit="batch", smoothing=0.0, dynamic_ncols=True,
        ):
            # sequence
            sequence_batch = hdf["sequence"].iloc[
                i*batch_size:(i+1)*batch_size
            ].to_list()
            # token
            token_batch = tokenizer(
                sequence_batch, return_tensors = 'pt', padding=True
            )["input_ids"].to(device)

            # chr
            chr_batch = int(c) if c != "X" else 23
            chr_batch = torch.ones(len(sequence_batch)) * chr_batch
            # pos
            pos_batch = torch.tensor(hdf["pos"].iloc[
                i*batch_size:(i+1)*batch_size
            ].to_numpy()).float()
            # coord
            coord_batch = torch.stack([chr_batch, pos_batch], dim=1).to(device)
            ## 23 chromosomes, 1-based
            coord_batch[:, 0] = (coord_batch[:, 0]-1) / 22  
            ## 2.5e8 bp in human genome
            coord_batch[:, 1] = coord_batch[:, 1] / 2.5e8

            # embedding
            with torch.no_grad():
                embedding_batch = model(
                    token_batch, coord_batch, embedding=True
                ).detach().cpu().numpy()
            # save
            embd = np.concatenate(
                [embd, embedding_batch], axis=0
            ) if embd is not None else embedding_batch
        embd: np.ndarray = np.concatenate([
            # [0:768] for embedding
            embd,
            # [768] for position                       
            hdf["pos"].to_numpy().reshape(-1, 1),
            # [769:776] for # of variants with p-value <= 1e-[0:7] cover by read
            np.concatenate([
                hdf[f"{_:.0e}"].to_numpy().reshape(-1, 1)
                for _ in pval_thresh_list
            ], axis=1, dtype=np.float32)
        ], axis=1, dtype=np.float32)
        embd = embd[embd[:, 768].argsort()]

        # bucket and bucket2pos
        bucket, bucket2pos = np.unique(embd[:,  768]//1000, return_index=True)
        bucket2pos = np.concatenate((bucket2pos, [len(embd)]))
        bucket = bucket.astype(int).tolist()
        bucket2pos = bucket2pos.astype(int).tolist()
        # store file in bucket
        for b in tqdm.tqdm(
            range(len(bucket)), leave=False,
            unit="bucket", desc=c, smoothing=0.0, dynamic_ncols=True,
        ):
            embd_bucket = embd[bucket2pos[b]:bucket2pos[b+1]]
            hash_idx = f"{bucket[b]:06d}"
            hash_fold, hash_file = hash_idx[:3], hash_idx[3:]+".npy"
            os.makedirs(
                os.path.join(embd_save_fold, c, hash_fold), exist_ok=True
            )
            np.save(
                os.path.join(embd_save_fold, c, hash_fold, hash_file), 
                embd_bucket
            )
