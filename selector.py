import torch

import numpy as np
import pandas as pd

import os
import tqdm


class Selector:
    def __init__(
        self, feature_fold: str, 
        pval_thresh: float = 1e-5, ascending: bool = False
    ) -> None:
        # path
        self.feature_fold = feature_fold
        self.profile_path = os.path.join(feature_fold, "profile.csv")
        if not os.path.exists(feature_fold): os.makedirs(feature_fold)
        # parameter
        self.pval_thresh = pval_thresh
        self.ascending = ascending

        self.profile = pd.DataFrame(columns=["embd_path", "feature_path"])
        if os.path.exists(self.profile_path): 
            self.profile = pd.read_csv(self.profile_path)

    def adds(self, embd_path_list: list[str]) -> None:
        for embd_path in embd_path_list: self.add(embd_path)

    def add(self, embd_path: str) -> None:
        # if embd_path already in profile, skip
        if embd_path in self.profile["embd_path"].values: return
        # if not, add to end of profile
        index = len(self)
        self.profile.loc[index, "embd_path"] = embd_path
        self.profile.loc[index, "feature_path"] = os.path.join(
            self.feature_fold, f"{index:04}.npy"
        )

        x1index = index
        x1embd, x1pos = self._getEmbd(x1index)
        for x2index in tqdm.tqdm(
            range(len(self)), desc=f"{x1index:04}", 
            dynamic_ncols=True, unit="run",
        ):
            x2embd, x2pos = self._getEmbd(x2index)
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

    def _getEmbd(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        # read embd of given path
        embd = np.load(self.profile.loc[index, "embd_path"])
        # filter read that cover at least one variants with p-value<=pval_thresh
        embd = embd[embd[:, 769-int(np.log10(self.pval_thresh))]>=1, :]
        # embd[:, :768] for embedding, embd[:, 768] for pos
        return embd[:, :768], embd[:, 768]  # (N, 768), (N,)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        feature = np.zeros([0, 2])
        if isinstance(index, slice):
            # concat selected features of all runs
            feature = np.zeros([0, 2])
            for i in range(*index.indices(len(self))):
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
        if isinstance(index, int) and index >= 0 and index < len(self):
            feature_path = self.profile.loc[index, "feature_path"]
            if os.path.exists(feature_path): feature = np.load(feature_path)
        return feature

    def __len__(self) -> int:
        return len(self.profile)


if __name__ == "__main__":
    feature_fold = "data/feature/"
    pval_thresh = 1e-5
    ascending = False   # False for top max distance, True for top min distance

    data_fold = "data/stanford/"
    method = "pretrain"     # pretrain or finetune
    chromosome = "6"
    
    selector = Selector(feature_fold, pval_thresh, ascending)
    selector.adds([
        os.path.join(data_fold, f"embd-{method}/{run}/{chromosome}.npy" )
        for run in os.listdir(os.path.join(data_fold, f"embd-{method}"))
    ])
