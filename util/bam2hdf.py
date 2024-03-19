import numpy as np
import pandas as pd

import os
import tqdm
import pysam
import argparse


__all__ = []


def main() -> None: 
    args = getArgs()
    bam2hdf(**vars(args))


def getArgs():
    parser = argparse.ArgumentParser(
        description="Go through reads in each chromosome of the BAM file. " + 
        "For each read, filter the read that is not paired, not properly " + 
        "paired, and not mapped. Then, cut bases with quality less than " + 
        "`quality_thresh=16` at beginning and end of reads and short reads " + 
        "with length less than `length_thresh=96`. Calculate num of variants " +
        "cover by each read with different p-value threshold. Save result as " +
        "a HDF5, seperate result of each chromosome by key `/chr{$chr}`, " + 
        "where `$chr` is 1-22 and X. Each chromosome result saves as " + 
        "dataframe with columns `sequence`, `pos`, `1e+00`, `1e-01`, " + 
        "`1e-02`, `1e-03`, `1e-04`, `1e-05`, and `1e-06`. For downstream " + 
        "analysis, load the HDF5 instead of BAM since BAM is hard to random " + 
        "index; we can only go through the whole BAM to filter reads, which " + 
        "is time-consuming. The algorithm can process 30000 reads per " + 
        "second, 1.5 hour for one sample, and need about 45GB memory. It's " + 
        "not optimized for parallel computing; open multiple terminals to " + 
        "process samples at the same time if memory is enough."
    )
    parser.add_argument(
        "-B", type=str, required=True, dest="bam_load_path",
        help="Path to load the BAM file. Sturcture `data_fold/bam/id.bam` is " +
        "recommended."
    )
    parser.add_argument(
        "-S", type=str, required=True, dest="snp_load_path",
        help="Path to load the SNPs file end with .tsv that contain genetic " + 
        "variation."
    )
    parser.add_argument(
        "-H", type=str, required=True, dest="hdf_save_path",
        help="Path to save the HDF5 file. Sturcture `data_fold/hdf/id.h5` is " +
        "recommended."
    )
    parser.add_argument(
        "-q", type=int, required=False, dest="quality_thresh", default=16,
        help="Cut low quality bases at beginning and end of reads. Default: 16."
    )
    parser.add_argument(
        "-l", type=int, required=False, dest="length_thresh", default=96,
        help="Skip reads with length less than this value. Default: 96.",
    )
    args = parser.parse_args()
    return args


def bam2hdf(
    bam_load_path: str, snp_load_path: str, hdf_save_path: str,
    quality_thresh: int = 16, length_thresh: int = 96,
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


if __name__ == "__main__": main()
