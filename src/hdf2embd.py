import torch
import transformers

import numpy as np
import pandas as pd

import os
import tqdm
import warnings
import argparse

from src.embd.model import PretrainModel, FinetuneModel


__all__ = ["hdf2embdArgs", "hdf2embd"]


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton*")


def hdf2embdArgs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-H", type=str, required=True, dest="hdf_load_path",
        help="Path to load the HDF5 file. Sturcture `data_fold/hdf/$id.h5` " + 
        "is recommended."
    )
    parser.add_argument(
        "-E", type=str, required=True, dest="embd_save_fold",
        help="Fold to save the embedding. Sturcture `data_fold/embd/$id` " + 
        "is recommended. Each chromosome's embedding will be saved under " + 
        "this fold seperately as `$chr.npy` , a N by 776 numpy array where N " +
        "for number of reads. For each read, `[0:768]` is the embedding, " + 
        "`[768]` is the pos, and `[769:776]` is num of variants cover by " + 
        "each read with different p-value threshold. Note that `[768:776]` " + 
        "directly copy from columns `pos`, `1e+00`, `1e-01`, `1e-02`, " + 
        "`1e-03`, `1e-04`, `1e-05`, and `1e-06` of HDF5 file " + 
        "`-H HDF_LOAD_PATH`. If `$chr.npy` already exists, check if the " + 
        "reads number of that chromosome in the HDF5 file after filtering is " +
        "larger than that in $chr.npy. If so, means the chromosome is not " + 
        "fully processed because the function run with a smaller p-value " + 
        "threshold before, and the function will reprocess the chromosome; " + 
        "if not, skip that chromosome."
    )
    parser.add_argument(
        "-C", type=str, required=False, dest="ckpt_load_path", default=None,
        help="Path to load checkpoint for `src.embd.model.FinetuneModel`. " + 
        "If not provided, `src.embd.model.PretrainModel` will be used. " + 
        "Default: None."
    )
    parser.add_argument(
        "-p", type=float, required=False, dest="pval_thresh", default=1e-04,
        help="P-value threshold for filtering reads, i.e., only keep reads " + 
        "that cover at least one variant with p-value <= pval_thresh. " + 
        "Default: 1e-04."
    )
    parser.add_argument(
        "-b", type=int, required=False, dest="batch_size", default=100,
        help="Batch size for number of reads input to tokenizer and model " + 
        "at same time. Default: 100."
    )


def hdf2embd(
    hdf_load_path: str, embd_save_fold: str, 
    ckpt_load_path: str = None,
    pval_thresh: float = 1e-04, batch_size: int = 100,
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

    for chr in tqdm.tqdm(
        [str(i) for i in range(1, 23)] + ["X"],     # BAM naming convention
        unit="chr", desc=embd_save_fold, smoothing=0.0, dynamic_ncols=True,
    ):
        embd_save_path = os.path.join(embd_save_fold, f"{chr}.npy")

        # get the hdf that store sequence and p-value for filtering
        hdf = pd.read_hdf(hdf_load_path, key=f"/chr{chr}", mode="r")
        hdf = hdf[hdf[f"{pval_thresh:.0e}"]>=1]

        # check if chromosome already processed
        if os.path.exists(embd_save_path):
            embd = np.load(embd_save_path)
            if len(embd) >= len(hdf):
                tqdm.tqdm.write(
                    f"{embd_save_path} already processed, skip."
                )
                continue

        embedding = None
        for i in tqdm.tqdm(
            range(int(np.ceil(len(hdf)/batch_size))), desc=chr, leave=False,
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
            chr_batch = int(chr) if chr != "X" else 23
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
            embedding = np.concatenate(
                [embedding, embedding_batch], axis=0
            ) if embedding is not None else embedding_batch

        # save
        result: np.ndarray = np.concatenate([
            # [0:768] for embedding
            embedding,
            # [768] for position                       
            hdf["pos"].to_numpy().reshape(-1, 1),
            # [769:776] for # of variants with p-value <= 1e-[0:7] cover by the read
            np.concatenate([
                hdf[f"{_:.0e}"].to_numpy().reshape(-1, 1)
                for _ in pval_thresh_list
            ], axis=1, dtype=np.float32)
        ], axis=1, dtype=np.float32)
        if not os.path.exists(embd_save_fold): os.makedirs(embd_save_fold)
        np.save(os.path.join(embd_save_fold, f"{chr}.npy"), result)
