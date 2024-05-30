import torch

import numpy as np
import scipy.spatial.distance

import os
import h5py
import tqdm
import functools
import multiprocessing


__all__ = ["Selector"]


class Selector:
    def __init__(self, feature_fold: str) -> None:
        self.feature_fold = feature_fold

    """ addFeature """

    def addFeatureChr(
        self, chromosome: str, chunk_size: int, bucket_size: int,
        embd_folds: str | list[str], 
        processes: int = None
    ) -> None:
        if isinstance(embd_folds, str): embd_folds = [embd_folds]
        for embd_fold in tqdm.tqdm(
            embd_folds, dynamic_ncols=True, smoothing=0, position=0,
            unit="sample", desc=f"chromosome {chromosome}"
        ):
            # embd and pos, sort by pos
            embd, pos = Selector.getEmbd(embd_fold, chromosome)
            # chunk index
            chunk, chunk2pos = np.unique(
                pos//(chunk_size*bucket_size), return_index=True
            )
            chunk2pos = np.append(chunk2pos, len(pos))
            # split pos and embd by chunk
            chunk_pos = [
                pos[chunk2pos[c]:chunk2pos[c+1]] for c in range(len(chunk))
            ]
            chunk_embd = [
                embd[chunk2pos[c]:chunk2pos[c+1]] for c in range(len(chunk))
            ]
            # check if directory of saving result exists
            os.makedirs(os.path.join(self.feature_fold, chromosome), exist_ok=1)
            # process each chunk
            jobs = [
                (chromosome, chunk[c], bucket_size, chunk_pos[c], chunk_embd[c]) 
                for c in range(len(chunk))
            ]
            with multiprocessing.Pool(processes=processes) as pool:
                list(tqdm.tqdm(
                    pool.imap(self.addFeatureChunkWrapper, jobs), 
                    total=len(jobs), dynamic_ncols=1, smoothing=0, leave=0,
                    desc=embd_fold, unit="chunk",
                ))

    def addFeatureChunkWrapper(self, args: tuple) -> None:
        self.addFeatureChunk(*args)

    def addFeatureChunk(
        self, chromosome: str, chunk: int, bucket_size: int,
        pos: np.ndarray, embd: np.ndarray,
    ) -> None:
        # bucket index
        bucket, bucket2pos = np.unique(pos//bucket_size, return_index=True)
        bucket2pos = np.append(bucket2pos, len(embd))
        # split embd by bucket
        bucket_embd = [
            embd[bucket2pos[b]:bucket2pos[b+1]] for b in range(len(bucket))
        ]
        # process each bucket
        for b in range(len(bucket)):
            self.addFeatureBucket(
                chromosome, chunk, bucket[b], bucket_embd[b]
            )

    def addFeatureBucket(
        self, chromosome: str, chunk: int, bucket: int, embd: np.ndarray
    ) -> None:
        path = os.path.join(self.feature_fold, chromosome, f"c{chunk:05}.hdf5")
        key  = f"b{bucket:08}"

        try:
            with h5py.File(path, 'a') as h5f:
                if key in h5f:
                    # get feature
                    feature = h5f[key]
                    feature_old = feature[:, :]
                    len_old = len(feature_old)
                    feature_new = np.concatenate([
                        feature_old, 
                        np.column_stack(
                            (embd, np.zeros(len(embd), dtype=np.float32))
                        )
                    ])
                    len_new = len(feature_new)                
                    # get max distance
                    dist = np.max(
                        #torch.cdist(
                        #    torch.from_numpy(embd), 
                        #    torch.from_numpy(feature[:, :768])
                        #).numpy(),
                        scipy.spatial.distance.cdist(
                            embd, feature_new[:, :768]
                        ), 
                        #Selector.getEuclideanDistance(embd, feature[:, :768]), 
                        axis=0
                    )
                    feature_new[:, 768] = dist
                    # save
                    feature.resize((len_new, 769))
                    feature[len_old:] = feature_new[len_old:]
                    feature[:len_old, 768] = feature_new[:len_old, 768]
                else:
                    # get feature
                    feature_new = np.column_stack(
                        (embd, np.zeros(len(embd), dtype=np.float32))
                    )
                    # get max distance
                    dist = np.max(
                        #torch.cdist(
                        #    torch.from_numpy(embd), 
                        #    torch.from_numpy(feature[:, :768])
                        #).numpy(),
                        scipy.spatial.distance.cdist(
                            embd, feature_new[:, :768]
                        ), 
                        #Selector.getEuclideanDistance(embd, feature[:, :768]), 
                        axis=0
                    )
                    feature_new[:, 768] = dist
                    # save
                    h5f.create_dataset(
                        key, data=feature_new, maxshape=(None, 769), chunks=True
                    )
        except (BlockingIOError, OSError) as _:
            print(f"{chromosome} c{chunk:05} b{bucket:08}")

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
        # sort row by position embd[:, 768]
        embd = embd[embd[:, 768].argsort()]
        # embd[:, :768] for embedding, embd[:, 768] for pos
        return embd[:, :768], embd[:, 768].astype(np.int32)     # (N, 768), (N,)

    def getEuclideanDistance(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1norm = np.sum(x1**2, axis=1, keepdims=True)           # (N1,  1)
        x2norm = np.sum(x2**2, axis=1, keepdims=True).T         # ( 1, N2)
        x1x2   = 2 * np.dot(x1, x2.T)                           # (N1, N2)
        return np.sqrt(np.maximum(0, x1norm + x2norm - x1x2))   # (N1, N2)

    @staticmethod
    def getPearsonCorrelation(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1centered = x1 - x1.mean(axis=1, keepdims=True)        # (N1, 768)
        x2centered = x2 - x2.mean(axis=1, keepdims=True)        # (N2, 768)
        numerator = np.dot(x1centered, x2centered.T)            # (N1, N2)
        x1var = np.sum(x1centered**2, axis=1, keepdims=True)    # (N1,  1)
        x2var = np.sum(x2centered**2, axis=1, keepdims=True)    # (N2,  1)
        denominator = np.sqrt(np.dot(x1var, x2var.T))           # (N1, N2)
        return numerator / denominator                          # (N1, N2)
