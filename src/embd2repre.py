import torch

import numpy as np
import pandas as pd
import sklearn.decomposition

import os
import tqdm
import pickle


__all__ = ["Selector"]


torch.set_num_threads(os.cpu_count())


class Selector:
    def __init__(self, feature_fold: str, bucket_size: int = 100) -> None:
        self.feature_fold = feature_fold
        if not os.path.exists(self.feature_fold): os.makedirs(self.feature_fold)
        # sample map
        # transformation between sample_idx and embd_fold 
        self.sample_map_path = os.path.join(self.feature_fold, "sample_map.csv")
        if os.path.exists(self.sample_map_path):
            self.sample_map = pd.read_csv(self.sample_map_path, index_col=0)
        else:
            self.sample_map = pd.DataFrame(columns=["embd_fold"])
        # chromosome
        self.chromosome_list = [str(i) for i in range(1, 23)] + ["X"]
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

    """ addFeature (train) """

    def addFeature(
        self, embd_fold: str | list[str], 
        chromosome: str, hash_idx_start: int = 0, hash_idx_end: int = 249000,
    ) -> None:
        """
        Example: to add feature for chromsome 1,
            >>> import torch
            >>> import pandas as pd
            >>> import os
            >>> from src.embd2repre import Selector
            >>> # sample to be added to the selector
            >>> profile = pd.read_csv("data/profile.csv")
            >>> profile = profile[profile["train"]==1]
            >>> embd_fold = profile["embd_fold"].to_list()
            >>> # selector
            >>> selector = Selector("data/feature")
            >>> # process chromosome 1 in 3 part parallel
            >>> # can be further parallel by divide hash_idx into more parts
            >>> selector.addFeature(
            ...     embd_fold, chromosome="1", 
            ...     hash_idx_start=     0, hash_idx_end= 80000
            ... )
            >>> selector.addFeature(
            ...     embd_fold, chromosome="1", 
            ...     hash_idx_start= 80000, hash_idx_end=160000
            ... )
            >>> selector.addFeature(
            ...     embd_fold, chromosome="1", 
            ...     hash_idx_start=160000
            ... )
        Example: process all chromosome in parallel with Slurm, refer to 
        [add.py](https://github.com/tianrui-qi/SIP-DB2/blob/c02edf434d6b78d03cc8f97a6b31b7917dbe67a1/add.py)
        and 
        [add.sh](https://github.com/tianrui-qi/SIP-DB2/blob/c02edf434d6b78d03cc8f97a6b31b7917dbe67a1/add.sh)
        """

        # input check
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
        sample_idx = [int(self.sample_map[
            self.sample_map["embd_fold"]==os.path.abspath(i)
        ].index[0]) for i in embd_fold]

        # loop thought hash_index since we use same hash_size for store feature
        # and embd of all sample
        hash_idx_start = max(hash_idx_start, 0)
        hash_idx_end = min(hash_idx_end, self.hash_idx_max[chromosome])
        for hash_idx in tqdm.tqdm(
            range(hash_idx_start, hash_idx_end), 
            dynamic_ncols=True, smoothing=0, 
            unit="hash", desc=f"addFeature/{chromosome}"
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
        new_embd = self.getEmbdByIdx(sample_idx, chromosome, hash_idx)
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
                embd = self.getEmbdByIdx(old_sample_idx, chromosome, hash_idx)
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
            new_dist, all_dist = Selector.calMaxEuclideanDist(
                new_embd[b][:, :768], 
                np.concatenate([old_embd[b][:, :768], new_embd[b][:, :768]])
            )
            all_dist[-len(new_dist):] = new_dist
            feature[b][:, 3] = np.maximum(feature[b][:, 3], all_dist)

        ## save feature
        feature = np.concatenate([feature[b] for b in feature.keys()])
        os.makedirs(os.path.dirname(feature_path), exist_ok=1)
        np.save(feature_path, feature)

    def getEmbdByIdx(
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
                embd_s, 
                np.full(len(embd_s), s, dtype=np.float32), 
                np.arange(len(embd_s), dtype=np.float32)
            ])
            # concat to embd
            embd = np.concatenate([embd, embd_s])
        # if no embd, means no sample cover this hash_idx, return
        if len(embd) == 0: return None

        # split embd by bucket_idx
        # { bucket_idx : (:768 embd, 768 pos, 769 sample_idx, 770 embd_idx) }
        bucket_idx = embd[:, 768] % self.hash_size // self.bucket_size
        return {int(b): embd[bucket_idx==b] for b in np.unique(bucket_idx)}

    """ getFeature """

    def getFeature(
        self, chromosome: str | list[str] = None
    ) -> dict[str, np.ndarray]:
        # input check
        if chromosome is None: chromosome = self.chromosome_list
        if isinstance(chromosome, str): chromosome = [chromosome]

        # get feature of each chromosome
        for c in chromosome:
            # check if feature of current chromosome already get
            feature_c_path = os.path.join(self.feature_fold, c, "feature.npy")
            if os.path.exists(feature_c_path): continue
            # get feature of current chromosome by loop through all hash_idx
            feature_c = []
            for hash_idx in tqdm.tqdm(
                range(self.hash_idx_max[c]), 
                dynamic_ncols=True, smoothing=0, 
                unit="hash", desc=f"getFeature/{c}",
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

    def applyFeature(
        self, embd_fold: str | list[str], chromosome: str, top_k: float = 0.15
    ) -> None:
        # input check
        if isinstance(embd_fold, str): embd_fold = [embd_fold]

        # get selected feature of chromosome
        # (hash_idx, bucket_idx, distance)
        feature_c = self.getFeature(chromosome)[chromosome]
        # select top_k percent of feature with largest distance
        top_k = int(len(feature_c) * top_k)
        order = np.sort(np.argsort(feature_c[:, 2])[-top_k:])
        feature_c = feature_c[order]
        # split by hash_idx
        feature_c = feature_c[:, :2].astype(int)
        hash_idx, hash2bucket = np.unique(feature_c[:, 0], return_index=True)
        hash_idx = hash_idx.tolist()
        hash2bucket = np.concatenate([hash2bucket, [len(feature_c)]])
        hash2bucket = {
            hash_idx[h]: feature_c[hash2bucket[h]:hash2bucket[h+1], 1].tolist()
            for h in range(len(hash_idx))
        }

        # apply feature for each sample
        # (:768 embd, 768 pos, 769 embd_idx)
        for e in range(len(embd_fold)):
            # check path
            if not isinstance(embd_fold[e], str): continue
            if not os.path.exists(embd_fold[e]): continue
            # loop through hash_idx to find feature of sample s
            feature_s = []
            for h in tqdm.tqdm(
                hash_idx, dynamic_ncols=True, smoothing=0,
                unit="hash", desc=f"applyFeature/{e}/{chromosome} "
            ):
                # { bucket_idx : (:768 embd, 768 pos, 769 embd_idx) }
                embd = self.getEmbdByFold(embd_fold[e], chromosome, h)
                for b in hash2bucket[h]:
                    if embd is None or b not in embd.keys():
                        feature_s.append(
                            np.full((1, 770), np.nan, dtype=np.float32)
                        )
                        continue
                    # choose the read that close to the center of the bucket
                    center = embd[b][:, 768]%self.hash_size%self.bucket_size
                    center = np.abs(center-(self.bucket_size/2))
                    center = np.argmin(center)
                    feature_s.append(embd[b][center])
            feature_s = np.vstack(feature_s, dtype=np.float32)
            # save
            feature_s_path = os.path.join(
                embd_fold[e], chromosome, "feature.npy"
            )
            os.makedirs(os.path.dirname(feature_s_path), exist_ok=1)
            np.save(feature_s_path, feature_s)

    def getEmbdByFold(
        self, embd_fold: str, chromosome: str, hash_idx: int
    ) -> dict[int, np.ndarray] | None:
        # load embd at hash_idx
        # (:768 embd, 768 pos, 769 embd_idx)
        hash_fold = f"{hash_idx:06d}.npy"[:3]
        hash_file = f"{hash_idx:06d}.npy"[3:]
        sample_path = os.path.join(embd_fold, chromosome, hash_fold, hash_file)
        if not os.path.exists(sample_path): return None
        embd = np.load(sample_path)[:, :769]    # (:768 embd, 768 pos)
        # add embd_idx to column 769
        embd = np.column_stack([embd, np.arange(len(embd), dtype=np.float32)])
        # if no embd, means no sample cover this hash_idx, return
        if len(embd) == 0: return None
        # split embd by bucket_idx
        # { bucket_idx : (:768 embd, 768 pos, 769 embd_idx) }
        bucket_idx = embd[:, 768] % self.hash_size // self.bucket_size
        return {int(b): embd[bucket_idx==b] for b in np.unique(bucket_idx)}

    """ getIPCA (train) """

    def getIPCA(
        self, embd_fold: str | list[str] = None, batch_size: int = 1
    ) -> sklearn.decomposition.IncrementalPCA:
        """
        Example: to use embd_fold for train to partial fit IncrementalPCA,
            >>> import pandas as pd
            >>> from src import Selector
            >>> profile = pd.read_csv("data/profile.csv")
            >>> profile = profile[profile["train"]==1]
            >>> embd_fold = profile["embd_fold"].to_list()
            >>> selector = Selector("data/feature")
            >>> ipca = selector.getIPCA(embd_fold, batch_size=40)
        Example: to load and use the fitted IncrementalPCA,
            >>> ipca = Selector("data/feature").getIPCA()
            >>> repre = ipac.transform(feature)
        """

        # input check
        if isinstance(embd_fold, str): embd_fold = [embd_fold]

        # load or init ipca
        ipca_path = os.path.join(self.feature_fold, "ipca.pkl")
        if os.path.exists(ipca_path):
            with open(ipca_path, "rb") as f: ipca = pickle.load(f)
        else: 
            ipca = sklearn.decomposition.IncrementalPCA(
                n_components=1, copy=False
            )
        if embd_fold is None: return ipca

        # partial fit ipca
        batch = 0
        features = []
        for e in tqdm.tqdm(
            embd_fold, dynamic_ncols=True, smoothing=0,
            unit="sample", desc="getIPCA",
        ):
            # load feature (:768 embd, 768 pos, 769 embd_idx)
            feature = np.vstack([
                np.load(os.path.join(e, c, "feature.npy"))
                for c in self.chromosome_list
            ])[:, :768].T                       # [768, K]
            features.append(feature)
            batch += 1
            if batch % batch_size != 0: continue
            # partial fit ipca
            features = np.vstack(features)      # [768 * batch_size, K]
            np.nan_to_num(features, nan=0.0, copy=False)
            ipca = ipca.partial_fit(features)
            features = []
        if len(features) != 0:
            features = np.vstack(features)      # [768 * batch_size, K]
            np.nan_to_num(features, nan=0.0, copy=False)
            ipca = ipca.partial_fit(features)
            features = []

        # save ipca
        os.makedirs(os.path.dirname(ipca_path), exist_ok=1)
        with open(ipca_path, "wb") as f: pickle.dump(ipca, f)
        return ipca

    """ getRepre """

    def getRepre(
        self, embd_fold: str | list[str], repre_path: str | list[str]
    ) -> None:
        """
        Example:
            >>> import pandas as pd
            >>> from src import Selector
            >>> profile = pd.read_csv("data/profile.csv")
            >>> embd_fold = profile["embd_fold"].to_list()
            >>> repre_path = profile["repre_path"].to_list()
            >>> Selector("data/feature").getRepre(embd_fold, repre_path)
        """

        # input check
        if isinstance(embd_fold, str): embd_fold = [embd_fold]
        if isinstance(repre_path, str): repre_path = [repre_path]
        assert len(embd_fold) == len(repre_path)

        ipca = self.getIPCA()
        for i in tqdm.tqdm(
            range(len(embd_fold)), dynamic_ncols=True, smoothing=0,
            unit="sample", desc="getRepre",
        ):
            # check path
            if not isinstance(embd_fold[i], str): continue
            if not isinstance(repre_path[i], str): continue
            if not os.path.exists(embd_fold[i]): continue
            # load feature (:768 embd, 768 pos, 769 embd_idx)
            feature = np.vstack([
                np.load(os.path.join(embd_fold[i], c, "feature.npy"))
                for c in self.chromosome_list
            ])[:, :768].T                       # [768, K]
            np.nan_to_num(feature, nan=0.0, copy=False)
            # transform
            repre = ipca.transform(feature)     # [768, 1]
            # save
            os.makedirs(os.path.dirname(repre_path[i]), exist_ok=1)
            np.save(repre_path[i], repre)

    """ help function """

    @staticmethod
    def calMaxEuclideanDist(
        x1: np.ndarray, x2: np.ndarray, batch_size: int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This function is equivalent to max of axis 0 of torch.cdist(x1, x2). 
        We use some tricks to avoid memory overflow: use torch.cdist to compute 
        batch_size of x1 and x2 each time, and directly get the max distance of 
        axis 0 of the batch distance matrix instead of save matrix in the 
        memory. 

        Args:
            x1: np.ndarray, shape (Ni, D)
            x2: np.ndarray, shape (Nj, D)
            batch_size: int, default 10000

        Returns:
            x1dist: np.ndarray, shape (Ni,)
                Max distance of axis 1 of Euclidean distance matrix between x1
                and x2
            x2dist: np.ndarray, shape (Nj,)
                Max distance of axis 0 of Euclidean distance matrix between x1 
                and x2

        Example:
            >>> import torch
            >>> import numpy as np
            >>> from src import Selector
            >>> x1 = np.random.rand(25000, 10)
            >>> x2 = np.random.rand(25000, 10)
            >>> # torch.cdist
            >>> dist_matrix = torch.cdist(
            ...     torch.from_numpy(x1), torch.from_numpy(x2)
            ... ).numpy()
            >>> x1dist_torch = np.max(dist_matrix, axis=1)
            >>> x2dist_torch = np.max(dist_matrix, axis=0)
            >>> # Selector.calMaxEuclideanDist
            >>> x1dist_our, x2dist_our = Selector.calMaxEuclideanDist(x1, x2)
            >>> # check if the results are the same
            >>> print(np.allclose(x1dist_torch, x1dist_our))
            True
            >>> print(np.allclose(x2dist_torch, x2dist_our))
            True
        """

        x1: torch.Tensor = torch.from_numpy(x1)     # (Ni, D)
        x2: torch.Tensor = torch.from_numpy(x2)     # (Nj, D)

        x1batchnum = (x1.shape[0] + batch_size - 1) // batch_size
        x2batchnum = (x2.shape[0] + batch_size - 1) // batch_size

        x1dist = torch.full((len(x1),), float('-inf'))
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
                # update the max distances for x1, x2 using the max distances of 
                # current batch
                x1dist[start_i:end_i] = torch.max(
                    x1dist[start_i:end_i], torch.max(dist_matrix, dim=1).values
                )
                x2dist[start_j:end_j] = torch.max(
                    x2dist[start_j:end_j], torch.max(dist_matrix, dim=0).values
                )
        return x1dist.numpy(), x2dist.numpy()

    @staticmethod
    def calPearsonCorrelation(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1centered = x1 - x1.mean(axis=1, keepdims=True)        # (N1, 768)
        x2centered = x2 - x2.mean(axis=1, keepdims=True)        # (N2, 768)
        numerator = np.dot(x1centered, x2centered.T)            # (N1, N2)
        x1var = np.sum(x1centered**2, axis=1, keepdims=True)    # (N1,  1)
        x2var = np.sum(x2centered**2, axis=1, keepdims=True)    # (N2,  1)
        denominator = np.sqrt(np.dot(x1var, x2var.T))           # (N1, N2)
        return numerator / denominator                          # (N1, N2)
