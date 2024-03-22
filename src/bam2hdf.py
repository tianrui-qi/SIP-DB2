import numpy as np
import pandas as pd

import os
import tqdm
import pysam
import argparse


__all__ = ["bam2hdfArgs", "bam2hdf"]


def bam2hdfArgs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-B", type=str, required=True, dest="bam_load_path",
        help="Path to load the BAM file. Sturcture `data_fold/bam/$id.bam` " + 
        "is recommended."
    )
    parser.add_argument(
        "-S", type=str, required=True, dest="snps_load_path",
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
    bam_load_path: str, snps_load_path: str, hdf_save_path: str,
    quality_thresh: int = 16, length_thresh: int = 96,
    *vargs, **kwargs
) -> None:
    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    chr_list = [str(i) for i in range(1, 23)] + ["X"]   # BAM naming convention

    snps_dict = {}
    snps_df = pd.read_csv(snps_load_path, sep="\t")
    for pval_thresh in pval_thresh_list:
        snps_dict[pval_thresh] = {}
        snps_df = snps_df[snps_df["Pval"] <= pval_thresh]
        for c in snps_df["Chr"].unique():
            # key, BAM naming convention
            chr = str(c) if c != 23 else "X"
            # value, one hot, 0/1 for non-variant/variant
            pos = snps_df[snps_df["Chr"] == c]["Pos"].values
            snps_dict[pval_thresh][chr] = np.zeros(pos.max()+1)
            snps_dict[pval_thresh][chr][pos] = 1
            # len(pos) may differ from snps_dict[pval_thresh][chr].sum()
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

        # bam file
        try:
            bam_file = pysam.AlignmentFile(bam_load_path, "rb").fetch(chr)
        except ValueError:
            bam_file = pysam.AlignmentFile(bam_load_path, "rb").fetch(f"chr{chr}")

        # dataframe with column "sequence", "pos", 
        # "1e+00", "1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06"
        read_dict = {
            key: [] for key in 
            ["sequence", "pos"] + 
            [f"{pval_thresh:.0e}" for pval_thresh in pval_thresh_list]
        }
        for read in tqdm.tqdm(
            bam_file, unit="read", 
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
                if chr not in snps_dict[pval_thresh]:
                    num = 0
                else:
                    num = np.sum(
                        snps_dict[pval_thresh][chr][pos:pos+len(sequence)]
                    )
                read_dict[f"{pval_thresh:.0e}"].append(num)

        # save
        if not os.path.exists(os.path.dirname(hdf_save_path)):
            os.makedirs(os.path.dirname(hdf_save_path))
        pd.DataFrame(read_dict).to_hdf(
            hdf_save_path, key=f"/chr{chr}", mode="a", format="f", index=False
        )
