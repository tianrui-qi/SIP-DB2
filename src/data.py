import torch
import torch.utils.data
from torch import Tensor

import os
import sys
if 'ipykernel' in sys.modules:
    import tqdm.notebook as tqdm
else:
    import tqdm     # since tqdm does not work in jupyter properly
import pysam
from typing import List, Tuple

__all__ = ["DNADataset"]


class DNADataset(torch.utils.data.Dataset):
    def __init__(
        self, bam_load_fold: str, sample_list: List[Tuple[str, int]],
        quality_threshold: float, length_threshold: int,
        **kwargs
    ) -> None:
        # path
        self.bam_load_fold = bam_load_fold
        self.sample_list = sample_list  # list of sample (id, label)
        # filter
        self.quality_threshold = quality_threshold
        self.length_threshold = length_threshold

        # data
        self.sequence_list, self.coord_list, self.label_list = self._setUp()

    def _setUp(self) -> Tuple[List[str], Tensor, Tensor]:
        sequence_list, coord_list, label_list = [], [], []

        # sample
        for (id, label) in tqdm.tqdm(
            self.sample_list, leave=False, dynamic_ncols=True, 
            unit="sample", desc="_setUp", 
        ):
            samfile = pysam.AlignmentFile(
                os.path.join(self.bam_load_fold, f"{id}.bam"), "r"
            )
            chr_list = list(samfile.references)[:25]    # 1-22, X, Y, M
            # chromosome
            for chr in tqdm.tqdm(
                range(len(chr_list)), leave=False, dynamic_ncols=True,
                unit="chr", desc=f"sample {id}", 
            ):
                # read
                for read in tqdm.tqdm(
                    samfile.fetch(chr_list[chr]), 
                    leave=False, dynamic_ncols=True,
                    unit="read", desc=f"chromosome {chr_list[chr]}", 
                ):
                    pos = read.reference_start  # 0-based
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
                    if len(sequence) < self.length_threshold: 
                        continue

                    # save
                    sequence_list.append(sequence)
                    coord_list.append([chr, pos])
                    label_list.append(label)
            samfile.close()

        coord_list = torch.tensor(coord_list, dtype=torch.float32)
        label_list = torch.tensor(label_list, dtype=torch.float32)

        # normalization
        coord_list[:, 0] /= 24      # 25 chromosomes, 0-based
        coord_list[:, 1] /= 2.5e8   # 2.5e8 bp in human genome

        return sequence_list, coord_list, label_list

    def __getitem__(self, index) -> Tuple[str, Tensor, Tensor]:
        index = torch.randint(0, len(self), (1,)).item()
        sequence = self.sequence_list[index]
        coord    = self.coord_list[index]
        label    = self.label_list[index]
        return sequence, coord, label

    def __len__(self) -> int:
        return len(self.sequence_list)
