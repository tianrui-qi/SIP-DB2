import torch

import numpy as np
import pandas as pd

import os
import tqdm


class Selector:
    def __init__(self, feature_fold: str) -> None:
        # path
        self.addFeature_fold = os.path.join(feature_fold, "addFeature")
        self.getFeature_fold = os.path.join(feature_fold, "getFeature")
        self.profile_path    = os.path.join(feature_fold, "profile.csv")
        if not os.path.exists(feature_fold): os.makedirs(feature_fold)

        # profile
        if os.path.exists(self.profile_path):
            self.profile = pd.read_csv(self.profile_path)
        else:
            self.profile = pd.DataFrame(columns=["embd_fold"])
            self.profile.to_csv(self.profile_path, index=False)

    """ addFeature """

    def addFeature(
        self, embd_fold: str | list[str], 
        pval_thresh: float = None, batch_size: int = 10000,
    ) -> None:
        # if embd_fold not in self.profile, add to end of self.profile
        # this step make sure input embd_fold can be indexed in self.profile
        # since index is used to store and track feature
        if isinstance(embd_fold, str): embd_fold = [embd_fold]
        for i in embd_fold:
            fold = os.path.abspath(i)
            if fold not in self.profile["embd_fold"].values:
                self.profile.loc[len(self.profile), "embd_fold"] = fold
        self.profile.to_csv(self.profile_path, index=False)

        # since all embd_fold can be indexed in self.profile, transfer input 
        # embd_fold from str fold path to index in self.profile
        embd_fold = [self.profile[
            self.profile["embd_fold"]==os.path.abspath(i)
        ].index[0] for i in embd_fold]

        # get all job need to be run (i, j, chromosome), i, j are sample index
        jobs = []
        for i in embd_fold:
            for j in self.profile.index[:i+1]:
                for chromosome in [str(i) for i in range(1, 23)] + ["X"]:
                    job = (i, j, chromosome) if i < j else (j, i, chromosome)
                    if job not in jobs: jobs.append(job)
        # since we use help function self.addFeatureJob to run each job 
        # independently, can be modified to parallel in the future
        for job in tqdm.tqdm(
            jobs, dynamic_ncols=True, smoothing=0, unit="job", desc="addFeature"
        ): self.addFeatureJob(*job, pval_thresh, batch_size)

    def addFeatureJob(
        self, i: int, j: int, chromosome: str,
        pval_thresh: float = None, batch_size: int = 10000,
    ) -> None:
        x1path = os.path.join(
            self.addFeature_fold, f"{i:08}/{j:08}/{chromosome}.npy"
        )
        x2path = os.path.join(
            self.addFeature_fold, f"{j:08}/{i:08}/{chromosome}.npy"
        )

        # check if feature already calculated
        if os.path.exists(x1path) and os.path.exists(x2path): return

        x1embd, x1pos = Selector.getEmbd(
            self.profile.loc[i, "embd_fold"], chromosome, pval_thresh
        )
        x2embd, x2pos = Selector.getEmbd(
            self.profile.loc[j, "embd_fold"], chromosome, pval_thresh
        )

        # max distance of x1 and x2
        x1, x2 = Selector.getMaxDist(x1embd, x2embd, batch_size)
        # combine pos and distance as feature
        x1 = np.column_stack((x1pos, x1))
        x2 = np.column_stack((x2pos, x2))
        # sort by distance
        order = np.argsort(x1[:, 1])
        x1 = x1[order[::-1]]
        order = np.argsort(x2[:, 1])
        x2 = x2[order[::-1]]
        # drop duplicates reads with same position, keep the first one
        _, index = np.unique(x1[:, 0], return_index=True)
        x1 = x1[np.sort(index)]
        _, index = np.unique(x2[:, 0], return_index=True)
        x2 = x2[np.sort(index)]

        # save
        if not os.path.exists(os.path.dirname(x1path)): 
            os.makedirs(os.path.dirname(x1path))
        if not os.path.exists(os.path.dirname(x2path)): 
            os.makedirs(os.path.dirname(x2path))
        np.save(x1path, x1)
        np.save(x2path, x2)

    @staticmethod
    def getEmbd(
        embd_fold: str, chromosome: str, pval_thresh: float = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # read embd of given path
        embd = np.load(os.path.join(embd_fold, f"{chromosome}.npy"))
        # filter read that cover at least one variants with p-value<=pval_thresh
        if pval_thresh is not None:
            embd = embd[embd[:, 769-int(np.log10(pval_thresh))]>=1, :]
        # embd[:, :768] for embedding, embd[:, 768] for pos
        return embd[:, :768], embd[:, 768]  # (N, 768), (N,)

    @staticmethod
    def getMaxDist(
        x1: np.ndarray, x2: np.ndarray, batch_size: int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function is equivalent to max of axis 1, 0 of torch.cdist(x1, x2). 
        We use some tricks to avoid memory overflow: use torch.cdist to compute 
        batch_size of x1 and x2 each time, and directly get the max distance of 
        axis 1 and 0 of the batch distance matrix instead of save matrix in the 
        memory. 

        To show the function equivalents to torch.cdist,
            x1 = np.random.rand(25000, 10)
            x2 = np.random.rand(25000, 10)
            # torch.cdist
            dist_matrix = torch.cdist(
                torch.from_numpy(x1), torch.from_numpy(x2)
            ).numpy()
            x1dist_torch = np.max(dist_matrix, axis=1)
            x2dist_torch = np.max(dist_matrix, axis=0)
            # Selector.getMaxDist
            x1dist_our, x2dist_our = getMaxDist(x1, x2)
            # check if the results are the same
            print(np.allclose(x1dist_torch, x1dist_our))
            print(np.allclose(x2dist_torch, x2dist_our))
        which will print True and True. 

        Args:
            x1: np.ndarray, shape (Ni, D)
            x2: np.ndarray, shape (Nj, D)
            batch_size: int, default 10000

        Returns:
            x1dist: np.ndarray, shape (Ni,), max distance of axis 1 of Euclidean 
                distance matrix between x1 and x2
            x2dist: np.ndarray, shape (Nj,), max distance of axis 0 of Euclidean 
                distance matrix between x1 and x2
        """

        x1: torch.Tensor = torch.from_numpy(x1)     # (Ni, D)
        x2: torch.Tensor = torch.from_numpy(x2)     # (Nj, D)

        x1dist = torch.full((len(x1),), float('-inf'))
        x2dist = torch.full((len(x2),), float('-inf'))

        x1batchnum = (x1.shape[0] + batch_size - 1) // batch_size
        x2batchnum = (x2.shape[0] + batch_size - 1) // batch_size

        for i in tqdm.tqdm(
            range(x1batchnum), leave=False, unit="batch", dynamic_ncols=True
        ):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, len(x1))
            x1_batch = x1[start_i:end_i]
            for j in tqdm.tqdm(
                range(x2batchnum), leave=False, unit="batch", dynamic_ncols=True
            ):
                start_j = j * batch_size
                end_j = min(start_j + batch_size, len(x2))
                x2_batch = x2[start_j:end_j]
                # compute the distance matrix between x1_batch and x2_batch
                dist_matrix = torch.cdist(x1_batch, x2_batch)
                # compute the max distance of current batch
                x1dist_curr = torch.max(dist_matrix, dim=1).values
                x2dist_curr = torch.max(dist_matrix, dim=0).values
                # update the max distances for x1 and x2
                x1dist[start_i:end_i] = torch.max(x1dist[start_i:end_i], x1dist_curr)
                x2dist[start_j:end_j] = torch.max(x2dist[start_j:end_j], x2dist_curr)

        return x1dist.numpy(), x2dist.numpy()

    """ getFeature """

    def getFeature(self, top_k: float = 0.1) -> dict[str, np.ndarray]:
        """
        Args:
            top_k: float, default 0.1. 
                If top_k <= 1, means top_k percentage of features of each sample
                in input sample index will be combined.
                If top_k > 1, means top_k number features of each sample in 
                input sample index will be combined.
        
        Returns:
            feature: dict[str, np.ndarray].
                Key is chromosome, from "1" to "22" and "X". 
                Value is feature of each chromosome, getting by function 
                self.getFeatureChr(chromosome, top_k). Check documentation of
                self.getFeatureChr for more details.
        """
        return {
            chromosome: self.getFeatureChr(chromosome, top_k)
            for chromosome in tqdm.tqdm(
                [str(i) for i in range(1, 23)] + ["X"], 
                dynamic_ncols=True, smoothing=0, 
                unit="chromosome", desc="getFeature"
            )
        }

    def getFeatureChr(self, chromosome: str, top_k: float = 0.1) -> np.ndarray:
        """
        Args:
            chromosome: str.
                The chromosome's feature we will get.
            top_k: float, default 0.1. 
                If top_k <= 1, means top_k percentage of features of each sample
                in input sample index will be combined.
                If top_k > 1, means top_k number features of each sample in 
                input sample index will be combined.

        Returns:
            feature_chromosome: np.ndarray, shape (N, 2).
                Feature of given chromosome, sort by position, by combining 
                top k percentage or number of each sample's features in 
                self.profile.
                - feature_chromosome[:, 0] is position 
                - feature_chromosome[:, 1] is distance
                - N is the feature number of given chromosome

        TODO: Now some of these features' pos are very close, which may cause
        problem. Need to solve either by bucketing; with a parameter bucket 
        width, choose the pos with largest distance in each bucket; each 
        bucket shall also have a distance of bucket width to prevent overlap.
        """

        # if feature already got before, direct return cached feature
        path = os.path.join(self.getFeature_fold, f"{chromosome}.npy")
        if os.path.exists(path): return np.load(path)

        feature_chromosome = np.zeros([0, 2])
        for i in self.profile.index:
            # get feature_i, (position, distance) of sample_i, sort by position
            # already removed duplcate position
            feature_i = None
            for j in self.profile.index:
                # feature_ij, the distance of each position of sample_i by 
                # distance matrix between sample_i and sample_j
                # calculate in self.addFeature, sort by distance, already 
                # removed duplicate position
                feature_path = os.path.join(
                    self.addFeature_fold, f"{i:08}/{j:08}/{chromosome}.npy"
                )
                if not os.path.exists(feature_path): continue
                feature_ij = np.load(feature_path)
                # sort feature_ij by position so that row/position of feature_ij
                # for all j are matched
                feature_ij = feature_ij[feature_ij[:,0].argsort()]
                # since each row match, we can directly use max distance between
                # feature_ij and feature_i to update the distance of feature_i
                if feature_i is None:
                    feature_i = feature_ij
                else:
                    feature_i = np.column_stack([
                        feature_i[:, 0], 
                        np.maximum(feature_i[:, 1], feature_ij[:, 1])
                    ])
            if feature_i is None: continue
            # sort by distance and add top_k of feature_i to feature_chromosome
            index = np.argsort(feature_i[:, 1])
            index = index[
                : -int(top_k*len(feature_i)) if top_k <= 1 else -int(top_k) : -1
            ]
            feature_chromosome = np.concatenate(
                [feature_chromosome, feature_i[index]]
            )
        # sort by distance
        order = np.argsort(feature_chromosome[:, 1])
        feature_chromosome = feature_chromosome[order[::-1]]
        # drop duplicates reads with same position, keep the first one
        _, index = np.unique(feature_chromosome[:, 0], return_index=True)
        feature_chromosome = feature_chromosome[np.sort(index)]
        # sort by position
        order = np.argsort(feature_chromosome[:, 0])
        feature_chromosome = feature_chromosome[order]

        # cache feature for future use
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.save(path, feature_chromosome)

        return feature_chromosome   # [N, 2]

    """
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
    """
