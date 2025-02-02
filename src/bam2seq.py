import numpy as np
import pandas as pd

import os
import tqdm
import pysam


__all__ = ["bam2seq"]


def bam2seq(
    bam_load_path: str, hdf_save_path: str, chromosome: str, 
    snps_load_path: str, quality_thresh: int = 16, length_thresh: int = 96,
    batch_size: int = 1e6, verbal: bool | int = True, *vargs, **kwargs
) -> None:
    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    chromosome_list = [str(i) for i in range(1, 23)] + ["X"]

    # if this sample and chromosome already processed, skip
    if os.path.exists(hdf_save_path):
        with pd.HDFStore(hdf_save_path, mode='r') as hdf:
            if f"/chr{chromosome}_batch0" in hdf.keys(): return

    dirname = os.path.dirname(hdf_save_path)
    if dirname and not os.path.exists(dirname): os.makedirs(dirname)

    # load snps
    snps_df = pd.read_hdf(snps_load_path, key=f"/chr{chromosome}")
    snps_dict = {}
    for pval_thresh in pval_thresh_list:
        # value, one hot, 0/1 for non-variant/variant
        pos = snps_df[snps_df["Pval"] < pval_thresh]["Pos"].values
        snps_dict[pval_thresh] = np.zeros(pos.max()+1)
        snps_dict[pval_thresh][pos] = 1
        # len(pos) may differ from snps_dict[pval_thresh].sum()
        # since pos may have duplicated values

    # bam file
    bam_file = pysam.AlignmentFile(bam_load_path, "rb")

    # dataframe with column "sequence", "pos", 
    # "1e+00", "1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06"
    read_dict = {
        key: [] for key in 
        ["sequence", "pos"] + 
        [f"{pval_thresh:.0e}" for pval_thresh in pval_thresh_list]
    }
    batch_index = 0
    for read in tqdm.tqdm(
        bam_file.fetch(
            bam_file.references[chromosome_list.index(chromosome)]
        ), 
        unit="read", desc=f"bam2seq:{hdf_save_path}:{chromosome}", 
        smoothing=0.0, dynamic_ncols=True, mininterval=1.0,
        disable=(not verbal) if isinstance(verbal, bool) else (0 > verbal),
    ):
        pos = read.reference_start
        sequence = read.query_sequence
        quality  = read.query_qualities

        # filter by flag
        if not read.is_paired: continue
        if not read.is_proper_pair: continue
        if read.is_unmapped: continue
        if read.mate_is_unmapped: continue

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

        # save to dictionary
        read_dict["sequence"].append(sequence)
        read_dict["pos"].append(pos)
        for pval_thresh in pval_thresh_list:
            # if current chromosome does not have any variant
            if pval_thresh not in snps_dict:
                num = 0
            else:
                num = np.sum(
                    snps_dict[pval_thresh][pos:pos+len(sequence)]
                )
            read_dict[f"{pval_thresh:.0e}"].append(num)

        # save to hdf if reach batch_size
        if len(read_dict["sequence"]) < batch_size: continue
        read_df = pd.DataFrame(read_dict)
        read_df.to_hdf(
            hdf_save_path, key=f"/chr{chromosome}_batch{batch_index}", 
            mode="a", format="table", index=False
        )
        read_dict = {
            key: [] for key in 
            ["sequence", "pos"] + 
            [f"{pval_thresh:.0e}" for pval_thresh in pval_thresh_list]
        }
        batch_index += 1
    # save remaining to hdf
    if len(read_dict["sequence"]) > 0:
        read_df = pd.DataFrame(read_dict)
        read_df.to_hdf(
            hdf_save_path, key=f"/chr{chromosome}_batch{batch_index}", 
            mode="a", format="table", index=False
        )

    # close bam file
    bam_file.close()
