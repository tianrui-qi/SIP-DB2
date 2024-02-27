from torch.utils.data import Dataset

import os
import tqdm
import pysam
import random
from typing import List, Tuple

__all__ = ["DNADataset"]


class DNADataset(Dataset):
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
        self.read_list = self._setUp()  # [(chr, position, sequence, label)]

    def _setUp(self) -> List[Tuple[int, int, str, int]]:
        read_list = []
        # sample
        for (id, label) in tqdm.tqdm(
            self.sample_list, leave=False, dynamic_ncols=True, 
            unit="sample", desc="_setUp", 
        ):
            samfile = pysam.AlignmentFile(
                os.path.join(self.bam_load_fold, f"{id}.bam"), "r"
            )
            chr_list = list(samfile.references)[:1]    # 1-22, X, Y, M
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
                    position = read.reference_start     # 0-based
                    sequence = read.query_sequence
                    quality  = read.query_qualities
            
                    # cut low quality bases at beginning and end
                    while len(quality) > 0 and \
                    quality[0] < self.quality_threshold:
                        sequence = sequence[1:]
                        quality = quality[1:]
                        position += 1
                    while len(quality) > 0 and \
                    quality[-1] < self.quality_threshold:
                        sequence = sequence[:-1]
                        quality = quality[:-1]

                    # skip short reads
                    if len(sequence) < self.length_threshold: 
                        continue

                    read_list.append((chr, position, sequence, label))
            samfile.close()
        return read_list

    def __getitem__(self, index) -> Tuple[int, int, str, int]:
        return random.choice(self.read_list)

    def __len__(self) -> int:
        return len(self.read_list)
