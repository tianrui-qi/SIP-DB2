import numpy as np
import pandas as pd

import os
import sys
import pysam
import argparse
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

__all__ = []


def bam2csv(
    bam_load_path: str, snp_load_path: str, csv_save_fold: str,
    quality_thresh: int = 16, length_thresh: int = 96,
) -> int:
    pval_thresh_list = [
        1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9
    ]
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

    for chr in tqdm.tqdm(chr_list, unit="chr"):
        read_dict = {key: [] for key in ["sequence", "pos"] + pval_thresh_list}
        for read in tqdm.tqdm(
            pysam.AlignmentFile(bam_load_path, "rb"), unit="read", 
            leave=False, smoothing=0.0
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
                sequence = sequence[1:]
                quality = quality[1:]
                pos += 1
            while len(quality) > 0 and \
            quality[-1] < quality_thresh:
                sequence = sequence[:-1]
                quality = quality[:-1]

            # filter by read length
            if len(sequence) < length_thresh: continue

            # save
            read_dict["sequence"].append(sequence)
            read_dict["chr"].append(chr)
            read_dict["pos"].append(pos)
            for pval_thresh in pval_thresh_list:
                # if current chromosome does not have any variant
                if chr not in snp_dict[pval_thresh]:
                    num = 0
                # if last variant of current chromosome is before current read
                elif pos+1 > len(snp_dict[pval_thresh][chr]):
                    num = 0
                else:
                    end = min(pos+len(sequence), len(snp_dict[pval_thresh][chr]))
                    num = snp_dict[pval_thresh][chr][pos:end].sum()
                read_dict[pval_thresh].append(num)
        if not os.path.exists(csv_save_fold): os.makedirs(csv_save_fold)
        pd.DataFrame(read_dict).to_csv(
            os.path.join(csv_save_fold, f"{chr}.csv"), index=False
        )


if __name__ == "__main__":
    bam2csv(
        bam_load_path = "data/bam/SRR8924593.bam",
        snp_load_path = "data/snp/snp_filter.tsv",
        csv_save_fold = "data/csv/SRR8924593/",
        quality_thresh = 16,
        length_thresh = 96
    )
