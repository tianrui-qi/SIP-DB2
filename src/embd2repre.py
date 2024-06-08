import torch

import numpy as np
import pandas as pd

import os
import tqdm


__all__ = ["Selector"]


class Selector:
    def __init__(self, feature_fold: str, bucket_size: int = 100) -> None:
        self.feature_fold = feature_fold
        if not os.path.exists(self.feature_fold): os.makedirs(self.feature_fold)
        # sample map, for transformation between sample_idx and embd_fold 
        self.sample_map_path = os.path.join(self.feature_fold, "sample_map.csv")
        if os.path.exists(self.sample_map_path):
            self.sample_map = pd.read_csv(self.sample_map_path, index_col=0)
        else:
            self.sample_map = pd.DataFrame(columns=["embd_fold"])
            self.sample_map.to_csv(self.sample_map_path)
        # hash
        self.hash_size = 1000
        self.hash_idx_max = {
             "1": 249000,  "2": 243000,  "3": 199000,  "4": 191000,
             "5": 182000,  "6": 171000,  "7": 160000,  "8": 146000,
             "9": 139000, "10": 134000, "11": 136000, "12": 134000,
            "13": 115000, "14": 108000, "15": 102000, "16":  91000,
            "17":  84000, "18":  81000, "19":  59000, "20":  65000,
            "21":  47000, "22":  51000,  "X": 157000,
        }
        # bucket
        self.bucket_size = bucket_size
        if self.hash_size % self.bucket_size != 0: raise ValueError(
            "hash_size {} must be divisible by bucket_size {}".format(
                self.hash_size, self.bucket_size
            )
        )

    """ addFeature """

    def addFeature(
        self, embd_fold: str | list[str], chromosome: str,
        hash_idx_start: int = 0, hash_idx_end: int = 249000,
    ) -> None:
        if isinstance(embd_fold, str): embd_fold = [embd_fold]

        # if embd_fold not in self.sample_map, add to end of self.sample_map
        # this step make sure embd_fold can be indexed in self.sample_map
        for i in embd_fold:
            fold = os.path.abspath(i)
            if fold not in self.sample_map["embd_fold"].values:
                self.sample_map.loc[len(self.sample_map), "embd_fold"] = fold
        self.sample_map.to_csv(self.sample_map_path)
        # since all embd_fold can be indexed in self.sample_map, transfer 
        # embd_fold from str fold path to index in self.sample_map
        sample_idx = [self.sample_map[
            self.sample_map["embd_fold"]==os.path.abspath(i)
        ].index[0] for i in embd_fold]

        # loop thought hash_index since we use same hash_size for store feature
        # and embd of all sample
        hash_idx_start = max(hash_idx_start, 0)
        hash_idx_end = min(hash_idx_end, self.hash_idx_max[chromosome])
        for hash_idx in tqdm.tqdm(
            range(hash_idx_start, hash_idx_end), 
            dynamic_ncols=True, smoothing=0, 
            unit="hash", desc=f"addFeature {chromosome}"
        ): self.addFeatureHash(sample_idx, chromosome, hash_idx)

    def addFeatureHash(
        self, sample_idx: int | list[int], chromosome: str, hash_idx: int
    ) -> None:
        hash_fold = f"{hash_idx:06d}.npy"[:3]
        hash_file = f"{hash_idx:06d}.npy"[3:]

        # embd, including new_embd and old_embd
        # { bucket_idx : (:768 embd, 768 pos, 769 sample_idx, 770 embd_idx) }
        # feature, including new_feature and old_feature)
        # { bucket_idx : (sample_idx, pos, embd_idx, distance) }

        ## new_embd/feature
        # load embd at hash_idx of all input sample_idx, split by bucket_idx
        new_embd = self.getEmbd(sample_idx, chromosome, hash_idx)
        if new_embd is None: return
        # create feature for new embd
        new_feature = {
            int(b): np.column_stack([
                new_embd[b][:, 769], new_embd[b][:, 768], new_embd[b][:, 770],
                np.zeros(len(new_embd[b]), dtype=np.float32)
            ]) for b in new_embd.keys()
        }

        ## old_embd/feature
        old_embd = {
            b: np.zeros((0, 768+3), dtype=np.float32) for b in new_embd.keys()
        }
        old_feature = {
            b: np.zeros((0, 4), dtype=np.float32) for b in new_embd.keys()
        }
        # load and update old_embd/feature if feature_path exists
        feature_path = os.path.join(
            self.feature_fold, chromosome, hash_fold, hash_file
        )
        if os.path.exists(feature_path): 
            # update old_feature
            feature = np.load(feature_path)
            bucket_idx = feature[:, 1] % self.hash_size // self.bucket_size
            for b in np.unique(bucket_idx):
                old_feature[int(b)] = feature[bucket_idx==b]
            # update old_embd
            # since we only want to update buckets that in new_embd.keys(),
            # get the sample_idx in old_feature that participate in the 
            # calculation of these buckets
            old_sample_idx = np.unique(
                feature[np.isin(bucket_idx, list(new_embd.keys()))][:, 0]
            ).astype(int).tolist()
            if len(old_sample_idx) != 0:
                embd = self.getEmbd(old_sample_idx, chromosome, hash_idx)
                for b in embd.keys(): old_embd[b] = embd[b]
                # sort old_embd/feature by sample_idx and embd_idx to match
                for b in new_embd.keys():
                    old_embd[b] = old_embd[b][
                        np.lexsort([old_embd[b][:, 770], old_embd[b][:, 769]])
                    ]
                    old_feature[b] = old_feature[b][
                        np.lexsort([old_feature[b][:, 2], old_feature[b][:, 0]])
                    ]

        ## merge old and new feature
        feature = old_feature.copy()
        for b in new_feature.keys():
            feature[b] = np.concatenate([feature[b], new_feature[b]])

        ## calculate max distance and update feature
        for b in new_embd.keys():
            dist = Selector.getMaxDist(
                new_embd[b][:, :768], 
                np.concatenate([old_embd[b][:, :768], new_embd[b][:, :768]])
            )
            feature[b][:, 3] = np.maximum(feature[b][:, 3], dist)

        ## save feature
        feature = np.concatenate([feature[b] for b in feature.keys()])
        os.makedirs(os.path.dirname(feature_path), exist_ok=1)
        np.save(feature_path, feature)

    def getEmbd(
        self, sample_idx: int | list[int], chromosome: str, hash_idx: int
    ) -> dict[int, np.ndarray] | None:
        if isinstance(sample_idx, int): sample_idx = [sample_idx]
        if len(sample_idx) == 0: return None

        hash_fold = f"{hash_idx:06d}.npy"[:3]
        hash_file = f"{hash_idx:06d}.npy"[3:]

        # load embd at hash_idx of all sample_idx
        # (:768 embd, 768 pos, 769 sample_idx, 770 embd_idx)
        embd = np.zeros((0, 768+3), dtype=np.float32)
        for s in sample_idx:
            sample_path = os.path.join(
                self.sample_map.loc[s, "embd_fold"], 
                chromosome, hash_fold, hash_file
            )
            if not os.path.exists(sample_path): continue
            embd_s = np.load(sample_path)[:, :768+1]    # (:768 embd, 768 pos)
            # add sample_idx to column 769, embd_idx to column 770
            embd_s = np.column_stack([
                embd_s, np.full(len(embd_s), s), np.arange(len(embd_s))
            ])
            # concat to embd
            embd = np.concatenate([embd, embd_s])

        # if no embd, means no sample cover this hash_idx, return
        if len(embd) == 0: return None

        # split embd by bucket_idx
        # { bucket_idx : (:768 embd, 768 pos, 769 sample_idx, 770 embd_idx) }
        bucket_idx = embd[:, 768] % self.hash_size // self.bucket_size
        return {int(b): embd[bucket_idx==b] for b in np.unique(bucket_idx)}

    @staticmethod
    def getMaxDist(
        x1: np.ndarray, x2: np.ndarray, batch_size: int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function is equivalent to max of axis 0 of torch.cdist(x1, x2). 
        We use some tricks to avoid memory overflow: use torch.cdist to compute 
        batch_size of x1 and x2 each time, and directly get the max distance of 
        axis 0 of the batch distance matrix instead of save matrix in the 
        memory. 

        To show the function equivalents to torch.cdist,
            x1 = np.random.rand(25000, 10)
            x2 = np.random.rand(25000, 10)
            # torch.cdist
            dist_matrix = torch.cdist(
                torch.from_numpy(x1), torch.from_numpy(x2)
            ).numpy()
            x2dist_torch = np.max(dist_matrix, axis=0)
            # Selector.getMaxDist
            x2dist_our = getMaxDist(x1, x2)
            # check if the results are the same
            print(np.allclose(x2dist_torch, x2dist_our))
        which will print True and True. 

        Args:
            x1: np.ndarray, shape (Ni, D)
            x2: np.ndarray, shape (Nj, D)
            batch_size: int, default 10000

        Returns:
            x2dist: np.ndarray, shape (Nj,)
                Max distance of axis 0 of Euclidean distance matrix between x1 
                and x2
        """

        x1: torch.Tensor = torch.from_numpy(x1)     # (Ni, D)
        x2: torch.Tensor = torch.from_numpy(x2)     # (Nj, D)

        x1batchnum = (x1.shape[0] + batch_size - 1) // batch_size
        x2batchnum = (x2.shape[0] + batch_size - 1) // batch_size

        x2dist = torch.full((len(x2),), float('-inf'))
        for i in range(x1batchnum):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, len(x1))
            x1_batch = x1[start_i:end_i]
            for j in range(x2batchnum):
                start_j = j * batch_size
                end_j = min(start_j + batch_size, len(x2))
                x2_batch = x2[start_j:end_j]
                # compute the distance matrix between x1_batch and x2_batch
                dist_matrix = torch.cdist(x1_batch, x2_batch)
                # update the max distances for x2 using the max distances of 
                # current batch
                x2dist[start_j:end_j] = torch.max(
                    x2dist[start_j:end_j], torch.max(dist_matrix, dim=0).values
                )
        return x2dist.numpy()

    """ getFeature """

    def getFeature(
        self, chromosome: str | list[str] = None
    ) -> dict[str, np.ndarray]:
        # input check
        if chromosome is None: 
            chromosome = [str(i) for i in range(1, 23)] + ["X"]
        if isinstance(chromosome, str): chromosome = [chromosome]

        # get feature of each chromosome
        for c in tqdm.tqdm(
            chromosome, dynamic_ncols=True, smoothing=0,
            unit="chromosome", desc="getFeature"
        ):
            # check if feature of current chromosome already get
            feature_c_path = os.path.join(self.feature_fold, c, "feature.npy")
            if os.path.exists(feature_c_path): continue
            # get feature of current chromosome by loop through all hash_idx
            feature_c = []
            for hash_idx in tqdm.tqdm(
                range(self.hash_idx_max[c]), 
                dynamic_ncols=True, leave=False, smoothing=0, 
                unit="hash", desc=c,
            ):
                # load feature
                hash_fold = f"{hash_idx:06d}.npy"[:3]
                hash_file = f"{hash_idx:06d}.npy"[3:]
                feature_ch_path = os.path.join(
                    self.feature_fold, c, hash_fold, hash_file
                )
                if not os.path.exists(feature_ch_path): continue
                feature_ch = np.load(feature_ch_path)
                # get max distance of each bucket
                bucket_idx = feature_ch[:, 1]%self.hash_size//self.bucket_size
                for b in np.unique(bucket_idx):
                    feature_c.append([
                        hash_idx, b, np.max(feature_ch[bucket_idx==b][:, 3])
                    ])
            # to ndarray and save
            feature_c = np.array(feature_c, dtype=np.float32)
            np.save(feature_c_path, feature_c)
        
        # { chromosome : (hash_idx, bucket_idx, distance) }
        return {
            c: np.load(os.path.join(self.feature_fold, c, "feature.npy"))
            for c in chromosome
        }

    """ applyFeature """

    """ getRepresentation """

    def getRepresentation():
        """
        TODO: run fit and transform
        """
        pass

    def fit():
        """
        TODO:
        1. check if pca and mean already stored in folder pca
        2. run applyFeature for sample in self.profile
        3. get the mean of all not nan part
        4. store the mean in folder pca for future use
        5. replace nan with mean
        6. run pca on all embd, store pca in folder pca for future use
        """
        pass

    def transform():
        """
        TODO:
        1. check if pca and mean already stored in folder pca
        2. run applyFeature for input embd_fold
        3. replace nan with mean and transform with pca
        """
        pass

    """ help function """

    @staticmethod
    def getPearsonCorrelation(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1centered = x1 - x1.mean(axis=1, keepdims=True)        # (N1, 768)
        x2centered = x2 - x2.mean(axis=1, keepdims=True)        # (N2, 768)
        numerator = np.dot(x1centered, x2centered.T)            # (N1, N2)
        x1var = np.sum(x1centered**2, axis=1, keepdims=True)    # (N1,  1)
        x2var = np.sum(x2centered**2, axis=1, keepdims=True)    # (N2,  1)
        denominator = np.sqrt(np.dot(x1var, x2var.T))           # (N1, N2)
        return numerator / denominator                          # (N1, N2)


class Other:
    def applyFeature(self, embd_fold: str, pos_range: int = 25) -> np.ndarray:
        """
        TODO: parallelize chromosome
        """
        return np.concatenate([
            self.applyFeatureChr(embd_fold, chromosome, pos_range)
            for chromosome in tqdm.tqdm(
                [str(i) for i in range(1, 23)] + ["X"],
                dynamic_ncols=True, smoothing=0, 
                unit="chromosome", desc="applyFeature"
            )
        ], axis=0)

    def applyFeatureChr(
        self, embd_fold: str, chromosome: str, pos_range: int = 25
    ) -> np.ndarray:
        """
        TODO: If embd_fold in self.profile, save the result in disk for future
        use of fit.
        """

        # feature of given chromosome, sort by position, 
        # only keep the position column
        path = os.path.join(self.getFeature_fold, f"{chromosome}.npy")
        if not os.path.exists(path): raise FileNotFoundError(
            f"Chromosome {chromosome} feature not found." + 
            "Please run getFeature or getFeatureChr first."
        )
        feature = self.getFeatureChr(chromosome)[:, 0]  # [Kc,]

        embd, pos = Selector.getEmbd(embd_fold, chromosome)
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

        return embd_selected    # [Kc, 768]
