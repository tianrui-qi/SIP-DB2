import torch

import numpy as np
import pandas as pd

import os
import tqdm
import matplotlib.pyplot as plt
from typing import Literal


class Selector:
    def __init__(self, feature_fold: str, ascending: bool = False) -> None:
        # path
        self.feature_fold = feature_fold
        self.profile_path = os.path.join(feature_fold, "profile.csv")
        if not os.path.exists(feature_fold): os.makedirs(feature_fold)
        # parameter
        self.ascending = ascending

        self.profile = pd.DataFrame(columns=["embd_path", "feature_path"])
        if os.path.exists(self.profile_path): 
            self.profile = pd.read_csv(self.profile_path)

    def add(self, embd_path: str, pval_thresh: float = None) -> None:
        # if embd_path already in profile, skip
        if embd_path in self.profile["embd_path"].values: return
        # if not, add to end of profile
        x1index = len(self.profile)
        self.profile.loc[x1index, "embd_path"] = embd_path
        self.profile.loc[x1index, "feature_path"] = os.path.join(
            self.feature_fold, f"{x1index:04}.npy"
        )

        x1embd, x1pos = Selector.getEmbd(embd_path, pval_thresh)
        for x2index in tqdm.tqdm(
            range(len(self.profile)), desc=f"{x1index:04}", 
            dynamic_ncols=True, unit="run", leave=False,
        ):
            x2embd, x2pos = Selector.getEmbd(
                self.profile.loc[x2index, "embd_path"], pval_thresh
            )
            # distance matrix between x1 and x2
            distance_matrix = torch.cdist(  # (Ni, Nj)
                torch.tensor(x1embd), torch.tensor(x2embd)
            ).numpy()
            # combine pos and distance as feature
            x1 = np.column_stack((x1pos, np.max(distance_matrix, axis=1)))
            x2 = np.column_stack((x2pos, np.max(distance_matrix, axis=0)))
            # if there are previous distance result, load and concate; we can 
            # direct concate since we only care about the max/min distance of 
            # each unique position; after sort and drop duplicates, the length
            # of distance result will not change
            x1 = np.concatenate([x1, self[x1index]], axis=0)
            x2 = np.concatenate([x2, self[x2index]], axis=0)
            # sort distance
            order = np.argsort(x1[:, 1])
            x1 = x1[order] if self.ascending else x1[order[::-1]]
            order = np.argsort(x2[:, 1])
            x2 = x2[order] if self.ascending else x2[order[::-1]]
            # drop duplicates reads with same position, keep the first one
            _, index = np.unique(x1[:, 0], return_index=True)
            x1 = x1[np.sort(index)]
            _, index = np.unique(x2[:, 0], return_index=True)
            x2 = x2[np.sort(index)]
            # save
            np.save(self.profile.loc[x1index, "feature_path"], x1)
            np.save(self.profile.loc[x2index, "feature_path"], x2)

        # save profile
        self.profile.to_csv(self.profile_path, index=False)

    def plot(self):
        feature = self[:]
        upper = int(np.ceil(feature[0, 1]))
        lower = int(np.floor(feature[-1, 1]))

        # 10%
        #upper_10 = feature[int(len(feature)*0.1), 1]
        #plt.axvline(upper_10, color="black", linestyle="--")
        #plt.text(upper_10+0.1, plt.ylim()[1]*3200, "10%")
        # 50%
        #upper_50 = feature[int(len(feature)*0.5), 1]
        #plt.axvline(upper_50, color="black", linestyle="--")
        #plt.text(upper_50+0.1, plt.ylim()[1]*3200, "50%")

        plt.hist(feature[:, 1], bins=10*(upper-lower), color="salmon")
        plt.xlim(lower, upper)
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Number of Features/Positions")
        plt.show()

    def apply(
        self, embd_path: str, pval_thresh: float = None, 
        top_k: float = 0.1, pos_range = 32, mod: Literal["all", "sep"] = "sep",
    ) -> np.ndarray:
        if mod not in ["all", "sep"]:
            raise ValueError("mod must be 'all' or 'sep'")
        elif mod == "all":
            # get top_k features from all run
            feature = self[:]
            if top_k <= 1: top_k = int(top_k * len(feature))     # percentage
            feature = feature[:top_k, 0]
        elif mod == "sep":
            # get top_k features for each run and combine
            feature = np.zeros([0])
            for i in range(len(self.profile)):
                feature_i = self[i]
                if top_k <= 1: top_k = int(top_k * len(feature_i))
                feature = np.concatenate([feature, feature_i[:top_k, 0]])
            # drop duplcates
            feature = np.unique(feature)
        # algorithm below assume feature is sorted by pos from small to large
        feature = np.sort(feature)

        embd, pos = Selector.getEmbd(embd_path, pval_thresh)
        embd_selected = np.full((len(feature), 768), np.nan)
        e = 0   # index for embd
        for p in range(len(feature)):
            # move e to the first read that is larger than feature[p]-pos_range
            # note that both pos and feature are sorted, so for next p, we can 
            # directly start from previous e
            while e < len(embd) and pos[e] < feature[p]-pos_range: e += 1
            if e >= len(embd): break
            # if first read that larger than feature[p]-pos_range is also larger 
            # than feature[p]+pos_range, then we have no read in this range, set 
            # the embd of this pos to nan; 
            if pos[e] > feature[p]+pos_range: continue
            # if first read that larger than feature[p]-pos_range smaller than 
            # feature[p]+pos_range, get the reads that closest to feature[p]
            e_temp = e
            distance = pos_range
            while e_temp < len(embd) and pos[e_temp] <= feature[p]+pos_range:
                if abs(pos[e_temp] - feature[p]) <= distance:
                    distance = abs(pos[e_temp] - feature[p])
                    embd_selected[p] = embd[e_temp, :]
                e_temp += 1
            if e >= len(embd): break
        return embd_selected    # [len(feature), 768]

    @staticmethod
    def getEmbd(
        embd_path: str, pval_thresh: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # read embd of given path
        embd = np.load(embd_path)
        # filter read that cover at least one variants with p-value<=pval_thresh
        if pval_thresh is not None:
            embd = embd[embd[:, 769-int(np.log10(pval_thresh))]>=1, :]
        # embd[:, :768] for embedding, embd[:, 768] for pos
        return embd[:, :768], embd[:, 768]  # (N, 768), (N,)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        # feature[:, 0] is pos, feture[:, 1] is distance, sort by distance
        feature = np.zeros([0, 2])
        if isinstance(index, slice):
            # concat selected features of all runs
            feature = np.zeros([0, 2])
            for i in range(*index.indices(len(self.profile))):
                feature_path = self.profile.loc[i, "feature_path"]
                if os.path.exists(feature_path): feature = np.concatenate(
                    [feature, np.load(feature_path)], axis=0
                )
            # sort distance
            order = np.argsort(feature[:, 1])
            feature = feature[order] if self.ascending else feature[order[::-1]]
            # drop duplicates reads with same position, keep the first one
            _, index = np.unique(feature[:, 0], return_index=True)
            feature = feature[np.sort(index)]
        if isinstance(index, int) and index >= 0 and index < len(self.profile):
            feature_path = self.profile.loc[index, "feature_path"]
            if os.path.exists(feature_path): feature = np.load(feature_path)

        # TODO: now some of these features' pos are very close, which may cause
        # problem. Need to solve either by bucketing; with a parameter bucket 
        # width, choose the pos with largest distance in each bucket; each 
        # bucket shall also have a distance of bucket width to prevent overlap.

        return feature

    def __len__(self) -> int:
        return len(self.profile)


if __name__ == "__main__":
    feature_fold = "data/feature/"
    ascending = False   # False for top max distance, True for top min distance

    data_fold = "data/stanford/"
    method = "pretrain"     # pretrain or finetune
    chr_list = [str(i) for i in range(1, 23)] + ["X"]   # BAM naming convention
    pval_thresh = 1e-5

    for chromosome in tqdm.tqdm(
        chr_list, 
        dynamic_ncols=True, unit="chr", 
    ):
        selector = Selector(os.path.join(feature_fold, chromosome), ascending)
        for run in tqdm.tqdm(
            os.listdir(os.path.join(data_fold, f"embd-{method}")),
            dynamic_ncols=True, unit="run", desc=chromosome, leave=False,
        ):
            selector.add(
                os.path.join(data_fold, f"embd-{method}/{run}/{chromosome}.npy"), 
                pval_thresh
            )
