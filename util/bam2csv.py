import numpy as np
import pandas as pd

import os
import sys
import pysam
import argparse
if 'ipykernel' in sys.modules:
    import tqdm.notebook as tqdm
else:
    import tqdm

__all__ = []


def main():
    args = getArgs()
    bam2csv(**vars(args))


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-B", type=str, required=True, dest="bam_load_path",
        help="Path to the BAM file. Sturcture of `data_fold/bam/id.bam` is " +
        "recommended."
    )
    parser.add_argument(
        "-S", type=str, required=False, dest="snp_load_path", default=None,
        help="Path to the SNPs file that contain genetic variation, " + 
        "optional. Default: assume `bam_load_path` has a structure of " + 
        "`data_fold/bam/*.bam`, load SNPs from `data_fold/snp/snp_filter.tsv`."
    )
    parser.add_argument(
        "-C", type=str, required=False, dest="csv_save_fold", default=None,
        help="Go through each chromosome's reads in BAM file and calculate " + 
        "num of variants in each read with different p-value threshold. Then " +
        "save each chromosome as seperate CSV file with header: " +
        "`sequence`, `pos`, `1e-0`, `1e-1`, `1e-2`, `1e-3`, `1e-4`. " + 
        "We only consider p-value to 1e-4 since the order of magnitude of" +
        "number of variants will not change a lots after 1e-4. Default: " +
        "assume `bam_load_path` has a structure of `data_fold/bam/id.bam`, " +
        "save csv to `data_fold/csv/id/chr.csv` where chr is 1-22 and X."
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

    # set default value
    data_fold = os.path.dirname(os.path.dirname(args.bam_load_path))
    if args.snp_load_path is None:
        args.snp_load_path = os.path.join(data_fold, "snp", "snp_filter.tsv")
    if args.csv_save_fold is None:
        csv_fold = os.path.join(data_fold, "csv")
        id = os.path.basename(args.bam_load_path).split(".")[0]
        args.csv_save_fold = os.path.join(csv_fold, id)

    return args


def bam2csv(
    bam_load_path: str, snp_load_path: str, csv_save_fold: str,
    quality_thresh: int = 16, length_thresh: int = 96,
) -> int:
    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, ]
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
        chr_list, unit="chr", desc=csv_save_fold,
        smoothing=0.0, dynamic_ncols=True,
    ):
        if not os.path.exists(csv_save_fold): os.makedirs(csv_save_fold)
        csv_save_path = os.path.join(csv_save_fold, f"{chr}.csv")

        # check if this chromosome already processed
        if os.path.exists(csv_save_path): continue

        read_dict = {key: [] for key in ["sequence", "pos"] + pval_thresh_list}
        for read in tqdm.tqdm(
            pysam.AlignmentFile(bam_load_path, "rb").fetch(chr), 
            unit="read", desc=csv_save_path, 
            leave=False, smoothing=0.0, dynamic_ncols=True, 
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
                read_dict[pval_thresh].append(num)

        # save
        pd.DataFrame(read_dict).to_csv(csv_save_path, index=False)


if __name__ == "__main__": main()
