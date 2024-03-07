import torch
import torch.utils.data
from torch import Tensor

import os
import pysam
import random
import pandas as pd
from typing import List, Tuple

__all__ = ["DNADataset"]


class DNADataset(torch.utils.data.Dataset):
    def __init__(
        self, num: int,
        snp_load_path: str, bam_load_fold: str, 
        sample_list: List[Tuple[str, int]],
        pos_range: int, quality_threshold: float, length_threshold: int,
        **kwargs
    ) -> None:
        self.num = num
        # path
        self.snp_load_path = snp_load_path
        self.bam_load_fold = bam_load_fold
        # sample
        self.sample_list = sample_list  # list of sample (id, label)
        # filter
        self.pos_range = pos_range
        self.quality_threshold = quality_threshold
        self.length_threshold  = length_threshold

        # data
        self.snp = pd.read_csv(snp_load_path, usecols=["Chr", "Pos"], sep="\t")
        self.sequence_list = []     # dna sequence of reads
        self.coord_list = []        # [chromosome, position] of reads
        self.label_list = []        # reads from pre or post anti-PD-1 sample

    def __getitem__(self, index) -> Tuple[str, Tensor, Tensor]:
        while len(self.sequence_list) == 0:
            # random choose a sample
            id, label = random.choice(self.sample_list)
            samfile = pysam.AlignmentFile(
                os.path.join(self.bam_load_fold, f"{id}.bam"), "r"
            )

            # random choose a bp at coord (chr_bp, pos_bp)
            chr_bp, pos_bp = self.snp.iloc[
                random.randint(0, len(self.snp)-1)
            ].values
            chr_list = [
                 '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', 
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                '21', '22', 'X'
            ]

            # fetch all read near the bp
            for read in samfile.fetch(
                contig=chr_list[chr_bp-1], 
                start=pos_bp-self.pos_range, end=pos_bp+self.pos_range
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
                quality[0] < self.quality_threshold:
                    sequence = sequence[1:]
                    quality = quality[1:]
                    pos += 1
                while len(quality) > 0 and \
                quality[-1] < self.quality_threshold:
                    sequence = sequence[:-1]
                    quality = quality[:-1]

                # filter by read length
                if len(sequence) < self.length_threshold: continue

                # save
                self.sequence_list.append(sequence)
                self.coord_list.append([chr_bp, pos])
                self.label_list.append(label)
            samfile.close()

        sequence = self.sequence_list.pop()
        coord = self.coord_list.pop()
        label = self.label_list.pop()

        # change data type
        coord = torch.as_tensor(coord, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.float32)
        # normalize
        coord[0] /= 22      # 23 chromosomes, 0-based
        coord[1] /= 2.5e8   # 2.5e8 bp in human genome

        return sequence, coord, label

    def __len__(self) -> int:
        return self.num
