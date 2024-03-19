import torch
import transformers

import numpy as np
import pandas as pd

import os
import tqdm
import pysam
import warnings
import argparse

import src


__all__ = []


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton*")


def main() -> None: 
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='func')
    bam2hdfArgs(subparsers.add_parser('bam2hdf'))
    hdf2embeddingArgs(subparsers.add_parser('hdf2embedding'))
    args = parser.parse_args()

    if args.func == "bam2hdf": 
        bam2hdf(**vars(args))
    if args.func == "hdf2embedding": 
        hdf2embedding(**vars(args))


def bam2hdfArgs(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-B", type=str, required=True, dest="bam_load_path",
        help="Path to load the BAM file. Sturcture `data_fold/bam/$id.bam` " + 
        "is recommended."
    )
    parser.add_argument(
        "-S", type=str, required=True, dest="snp_load_path",
        help="Path to load the SNPs file end with .tsv that contain genetic " + 
        "variation."
    )
    parser.add_argument(
        "-H", type=str, required=True, dest="hdf_save_path",
        help="Path to save the HDF5 file. Sturcture `data_fold/hdf/$id.h5` " + 
        "is recommended."
    )
    parser.add_argument(
        "-q", type=int, required=False, dest="quality_thresh", default=16,
        help="Cut low quality bases at beginning and end of reads. Default: 16."
    )
    parser.add_argument(
        "-l", type=int, required=False, dest="length_thresh", default=96,
        help="Skip reads with length less than this value. Default: 96.",
    )


def bam2hdf(
    bam_load_path: str, snp_load_path: str, hdf_save_path: str,
    quality_thresh: int = 16, length_thresh: int = 96,
    *vargs, **kwargs
) -> None:
    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    chr_list = [str(i) for i in range(1, 23)] + ["X"]   # BAM naming convention

    snp_dict = {}
    snp_df = pd.read_csv(snp_load_path, sep="\t")
    for pval_thresh in pval_thresh_list:
        snp_dict[pval_thresh] = {}
        snp_df = snp_df[snp_df["Pval"] <= pval_thresh]
        for c in snp_df["Chr"].unique():
            # key, BAM naming convention
            chr = str(c) if c != 23 else "X"
            # value, one hot, 0/1 for non-variant/variant
            pos = snp_df[snp_df["Chr"] == c]["Pos"].values
            snp_dict[pval_thresh][chr] = np.zeros(pos.max()+1)
            snp_dict[pval_thresh][chr][pos] = 1
            # len(pos) may differ from snp_dict[pval_thresh][chr].sum()
            # since pos may have duplicated values

    for chr in tqdm.tqdm(
        chr_list, unit="chr", 
        desc=hdf_save_path, smoothing=0.0, dynamic_ncols=True,
    ):
        # if this chromosome already processed, skip
        if os.path.exists(hdf_save_path):
            with pd.HDFStore(hdf_save_path, mode='r') as hdf:
                if f"/chr{chr}" in hdf.keys():
                    tqdm.tqdm.write(
                        f"{hdf_save_path}/chr{chr} already processed, skip."
                    )
                    continue

        # dataframe with column "sequence", "pos", 
        # "1e+00", "1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06"
        read_dict = {
            key: [] for key in 
            ["sequence", "pos"] + 
            [f"{pval_thresh:.0e}" for pval_thresh in pval_thresh_list]
        }
        for read in tqdm.tqdm(
            pysam.AlignmentFile(bam_load_path, "rb").fetch(chr), unit="read", 
            desc=chr, leave=False, smoothing=0.0, dynamic_ncols=True, 
        ):
            pos = read.reference_start
            sequence = read.query_sequence
            quality  = read.query_qualities

            # filter by flag
            if not read.is_paired: continue
            if not read.is_proper_pair: continue
            if read.is_unmapped: continue
            #if read.mate_is_unmapped: continue

            # cut low quality bases at beginning and end
            while len(quality) > 0 and \
            quality[0] < quality_thresh:
                sequence, quality = sequence[1:], quality[1:]
                pos += 1
            while len(quality) > 0 and \
            quality[-1] < quality_thresh:
                sequence, quality = sequence[:-1], quality[:-1]

            # filter by read length
            if len(sequence) < length_thresh: continue

            # save
            read_dict["sequence"].append(sequence)
            read_dict["pos"].append(pos)
            for pval_thresh in pval_thresh_list:
                # if current chromosome does not have any variant
                if chr not in snp_dict[pval_thresh]:
                    num = 0
                else:
                    num = np.sum(
                        snp_dict[pval_thresh][chr][pos:pos+len(sequence)]
                    )
                read_dict[f"{pval_thresh:.0e}"].append(num)

        # save
        if not os.path.exists(os.path.dirname(hdf_save_path)):
            os.makedirs(os.path.dirname(hdf_save_path))
        pd.DataFrame(read_dict).to_hdf(
            hdf_save_path, key=f"/chr{chr}", mode="a", format="f", index=False
        )


def hdf2embeddingArgs(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-H", type=str, required=True, dest="hdf_load_path",
        help="Path to load the HDF5 file. Sturcture `data_fold/hdf/$id.h5` " + 
        "is recommended."
    )
    parser.add_argument(
        "-N", type=str, required=True, dest="npy_save_fold",
        help="Fold to save the embedding. Sturcture `data_fold/$method/$id` " + 
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
        help="Path to load checkpoint for `src.FinetuneModel`. " + 
        "If not provided, `src.PretrainModel` will be used. Default: None."
    )
    parser.add_argument(
        "-p", type=float, required=False, dest="pval_thresh", default=1e-03,
        help="P-value threshold for filtering reads, i.e., only keep reads " + 
        "that cover at least one variant with p-value <= pval_thresh. " + 
        "Default: 1e-03."
    )
    parser.add_argument(
        "-b", type=int, required=False, dest="batch_size", default=100,
        help="Batch size for number of reads input to tokenizer and model " + 
        "at same time. Default: 100."
    )


def hdf2embedding(
    hdf_load_path: str, npy_save_fold: str, 
    ckpt_load_path: str = None,
    pval_thresh: float = 1e-03, batch_size: int = 100,
    *vargs, **kwargs
):
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    if ckpt_load_path is None:
        model = src.PretrainModel()
    else:
        ckpt = torch.load(ckpt_load_path)
        model = src.FinetuneModel(
            feats_token=ckpt["feats_token"], feats_coord=ckpt["feats_coord"], 
            feats_final=ckpt["feats_final"]
        )
        model.fc_token.load_state_dict(ckpt['fc_token'])
        model.fc_coord.load_state_dict(ckpt['fc_coord'])
    model.eval().to(device)

    chr_list = [str(i) for i in range(1, 23)] + ["X"]   # BAM naming convention
    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for chr in tqdm.tqdm(
        chr_list, unit="chr", 
        desc=npy_save_fold, smoothing=0.0, dynamic_ncols=True,
    ):
        npy_save_path = os.path.join(npy_save_fold, f"{chr}.npy")

        # get the hdf that store sequence and p-value for filtering
        hdf = pd.read_hdf(hdf_load_path, key=f"/chr{chr}", mode="r")
        hdf = hdf[hdf[f"{pval_thresh:.0e}"]>=1]

        # check if chromosome already processed
        if os.path.exists(npy_save_path):
            npy = np.load(npy_save_path)
            if len(npy) >= len(hdf):
                tqdm.tqdm.write(
                    f"{npy_save_path} already processed, skip."
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
        if not os.path.exists(npy_save_fold): os.makedirs(npy_save_fold)
        np.save(os.path.join(npy_save_fold, f"{chr}.npy"), result)


if __name__ == "__main__": main()
