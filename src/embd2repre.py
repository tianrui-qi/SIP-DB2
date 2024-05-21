import torch

import numpy as np
import pandas as pd

import os
import tqdm
import functools
import multiprocessing


__all__ = ["Selector"]


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
        self, embd_fold: str | list[str], batch_size: int = 10000,
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
                    # check if feature already calculated
                    x1path = os.path.join(
                        self.addFeature_fold, f"{i:08}/{j:08}/{chromosome}.npy"
                    )
                    x2path = os.path.join(
                        self.addFeature_fold, f"{j:08}/{i:08}/{chromosome}.npy"
                    )
                    if os.path.exists(x1path) and os.path.exists(x2path): 
                        continue
                    # add to jobs list if not in jobs list
                    job = (i, j, chromosome) if i < j else (j, i, chromosome)
                    if job not in jobs: jobs.append(job)
        # since we use help function self.addFeatureJob to run each job 
        # independently, can be modified to parallel in the future
        for job in tqdm.tqdm(
            jobs, dynamic_ncols=True, smoothing=0, unit="job", desc="addFeature"
        ): self.addFeatureJob(*job, batch_size)

    def addFeatureJob(
        self, i: int, j: int, chromosome: str, batch_size: int = 10000,
    ) -> None:
        x1path = os.path.join(
            self.addFeature_fold, f"{i:08}/{j:08}/{chromosome}.npy"
        )
        x2path = os.path.join(
            self.addFeature_fold, f"{j:08}/{i:08}/{chromosome}.npy"
        )

        # check if feature already calculated
        if os.path.exists(x1path) and os.path.exists(x2path): return

        x1fold = self.profile.loc[i, "embd_fold"]
        x1embd, x1pos = Selector.getEmbd(x1fold, chromosome)
        x2fold = self.profile.loc[j, "embd_fold"]
        x2embd, x2pos = Selector.getEmbd(x2fold, chromosome)

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

    """ getFeature """

    def getFeature(
        self, bin_size: int = 100, top_k: float = 0.1
    ) -> dict[str, np.ndarray]:
        # get feature for each chromosome and save in disk by calling
        # self.getFeatureChr in multiprocessing
        with multiprocessing.Pool() as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(
                    functools.partial(
                        self.getFeatureChr, bin_size=bin_size, top_k=top_k
                    ), [str(i) for i in range(1, 23)] + ["X"]
                ),
                total=23, dynamic_ncols=True, smoothing=0, 
                unit="chromosome", desc="getFeature"
            ): pass
        # return all chromosome feature as dict by load from disk
        return {
            chromosome: self.getFeatureChrCached(chromosome, top_k, verbose=1) 
            for chromosome in [str(i) for i in range(1, 23)] + ["X"]
        }

    def getFeatureChr(
        self, chromosome: str, bin_size: int = 100, top_k: float = 0.1
    ) -> np.ndarray:
        # if feature already got before, return feature that only keep top_k 
        # percentage of features
        path = os.path.join(self.getFeature_fold, f"{chromosome}.npy")
        if os.path.exists(path): 
            return self.getFeatureChrCached(chromosome, top_k)

        feature = np.zeros([0, 2])  # (position, distance)
        for i in self.profile.index:
            for j in self.profile.index:
                # feature_ij, the distance of each position of sample_i by 
                # distance matrix between sample_i and sample_j
                # calculate in self.addFeature, sort by distance, already 
                # removed duplicate position
                feature_ij_path = os.path.join(
                    self.addFeature_fold, f"{i:08}/{j:08}/{chromosome}.npy"
                )
                if not os.path.exists(feature_ij_path): continue
                feature_ij = np.load(feature_ij_path)

                # check if bucket size enough, extend if not
                bins_new = int(max(feature_ij[:, 0]) // bin_size) + 1
                bins_cur = len(feature)
                if bins_new > bins_cur:
                    extend = np.zeros((bins_new - bins_cur, 2))
                    extend[:, 0] = (np.arange(bins_cur, bins_new)+0.5)*bin_size
                    feature = np.concatenate([feature, extend])

                # let the third column of feature_ij be the bucket index
                feature_ij = np.column_stack([
                    feature_ij, (feature_ij[:, 0] // bin_size)
                ])
                # remove duplicate bucket_index and keep the first one
                # note that feature_ij sorted by distance, so by keep the first
                # one, we keep the row with largest distance for each bucket
                _, unique_index = np.unique(feature_ij[:, 2], return_index=True)
                feature_ij = feature_ij[unique_index]
                # for each feature, put it into corresponding bucket if that 
                # feature's distance is larger than the distance store in bucket
                feature_ij, bucket_index = feature_ij[:, 0:2], feature_ij[:, 2]
                bucket_index = bucket_index.astype(int)
                valid = feature_ij[:, 1] > feature[bucket_index, 1]
                feature[bucket_index[valid]] = feature_ij[valid]

        # cache feature for future use
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, feature)

        # return feature that only keep top_k percentage of features
        return self.getFeatureChrCached(chromosome, top_k)

    def getFeatureChrCached(
        self, chromosome: str, top_k: float = 0.1, verbose: bool = False,
    ) -> np.ndarray:
        # load cached feature
        path = os.path.join(self.getFeature_fold, f"{chromosome}.npy")
        if not os.path.exists(path): raise FileNotFoundError(
            f"Chromosome {chromosome} feature not found." + 
            "Please run getFeature or getFeatureChr first."
        )
        feature = np.load(path)
        feature_total = len(feature)    # record
        # transfer top_k into number of features we want to keep
        top_k = int(top_k*len(feature)) if top_k <= 1 else int(top_k)
        # remove row in feature that distance is 0
        feature = feature[feature[:, 1] != 0]
        feature_nonzero = len(feature)  # record
        # sort by distance
        feature = feature[np.argsort(feature[:, 1])[::-1]]
        feature_min = feature[0, 1]     # record
        feature_max = feature[0, 1]     # record
        # keep top_k percentage or number of features
        feature = feature[:top_k]
        # sort by position
        feature = feature[np.argsort(feature[:, 0])]

        if verbose: print(
            "Chromosome {}\t".format(chromosome) +
            f"min {feature_min:.2f} max {feature_max:.2f}\t" + 
            f"total {feature_total:>{10},}\t" +
            f"nonzero {feature_nonzero:>{10},}\t" +
            f"selected {len(feature):>{10},}\t" + 
            f"nonzero/total {feature_nonzero*100/feature_total:.2f}%\t" + 
            f"selected/total {len(feature)*100/feature_total:.2f}%\t" +
            f"selected/nonzero {len(feature)*100/feature_nonzero:.2f}%"
        )

        return feature  # [K, (position, distance)]

    """ applyFeature """

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
            x1dist: np.ndarray, shape (Ni,)
                Max distance of axis 1 of Euclidean distance matrix between x1 
                and x2
            x2dist: np.ndarray, shape (Nj,)
                Max distance of axis 0 of Euclidean distance matrix between x1 
                and x2
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x1: torch.Tensor = torch.from_numpy(x1).to(device)     # (Ni, D)
        x2: torch.Tensor = torch.from_numpy(x2).to(device)     # (Nj, D)

        x1dist = torch.full((len(x1),), float('-inf'), device=device)
        x2dist = torch.full((len(x2),), float('-inf'), device=device)

        x1batchnum = (x1.shape[0] + batch_size - 1) // batch_size
        x2batchnum = (x2.shape[0] + batch_size - 1) // batch_size

        for i in tqdm.tqdm(
            range(x1batchnum), leave=False, unit="batch", dynamic_ncols=True
        ):
            start_i = i * batch_size
            end_i = min(start_i + batch_size, len(x1))
            x1_batch = x1[start_i:end_i]
            for j in range(x2batchnum):
                start_j = j * batch_size
                end_j = min(start_j + batch_size, len(x2))
                x2_batch = x2[start_j:end_j]
                # compute the distance matrix between x1_batch and x2_batch
                dist_matrix = torch.cdist(x1_batch, x2_batch)
                # update the max distances for x1 and x2 using the max distances
                # of current batch
                x1dist[start_i:end_i] = torch.max(
                    x1dist[start_i:end_i], torch.max(dist_matrix, dim=1).values
                )
                x2dist[start_j:end_j] = torch.max(
                    x2dist[start_j:end_j], torch.max(dist_matrix, dim=0).values
                )

        return x1dist.cpu().numpy(), x2dist.cpu().numpy()

    @staticmethod
    def getPearsonCorrelation(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1centered = x1 - x1.mean(axis=1, keepdims=True)        # (N1, 768)
        x2centered = x2 - x2.mean(axis=1, keepdims=True)        # (N2, 768)
        numerator = np.dot(x1centered, x2centered.T)            # (N1, N2)
        x1var = np.sum(x1centered**2, axis=1, keepdims=True)    # (N1,  1)
        x2var = np.sum(x2centered**2, axis=1, keepdims=True)    # (N2,  1)
        denominator = np.sqrt(np.dot(x1var, x2var.T))           # (N1, N2)
        return numerator / denominator                          # (N1, N2)
