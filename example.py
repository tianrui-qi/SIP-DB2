import numpy as np
import pandas as pd
import umap
import sklearn.decomposition
import sklearn.svm, sklearn.linear_model
import sklearn.metrics, sklearn.preprocessing

import os
import tqdm
import pysam
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

import shutil
from src.embd2repre import Selector

verbal = True
skip_feature_selection = True

sample_name = ["a", "b", "c"]
bam_path    = [f"temp/bam/{i}.bam"   for i in sample_name]
hdf_path    = [f"temp/hdf/{i}.h5"    for i in sample_name]
embd_fold   = [f"temp/embd/{i}"      for i in sample_name]
repre_path  = [f"temp/repre/{i}.npy" for i in sample_name]
feature_fold = "temp/feature/"

snps_path = "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/snps.csv"

## step 0 ample Preparation ------------------------------
def make_sample(bam_load_path, bam_save_path, num_reads=100):
    # file check
    os.makedirs(os.path.dirname(bam_save_path), exist_ok=True)
    if os.path.exists(bam_save_path): os.remove(bam_save_path)
    if os.path.exists(bam_save_path + ".bai"): os.remove(bam_save_path + ".bai")
    # make sample
    input = pysam.AlignmentFile(bam_load_path, "rb")
    output = pysam.AlignmentFile(bam_save_path, "wb", template=input)
    for c in [str(i) for i in range(1, 23)] + ["X"]:
        count = 0
        for read in input.fetch(c):
            if count >= num_reads: break
            output.write(read)
            count += 1
    input.close()
    output.close()
    _ = pysam.sort("-o", bam_save_path, bam_save_path)
    _ = pysam.index(bam_save_path)
# make example samples, run once 
if not os.path.exists(bam_path[0]):
    make_sample("/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/bam/SRR8924580.bam", bam_path[0])
if not os.path.exists(bam_path[1]):
    make_sample("/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/bam/SRR8924581.bam", bam_path[1])
if not os.path.exists(bam_path[2]):
    make_sample("/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/bam/SRR8924582.bam", bam_path[2])

## step 1 BAM and SNPs to Sequence --------------------
# run command below in terminal
# can be run in parallel since bam2seq designed to be sample level parallel
# -q quality threshold
# -l quality read length
for i in range(len(bam_path)):
    if os.path.exists(hdf_path[i]): continue
    print("run step1...")
    cmd = f"python main.py bam2seq -B {bam_path[i]} -S {snps_path} -H {hdf_path[i]} -q {16} -l {96}"
    subprocess.run(cmd, shell=True)

# step 2 Run Embedding ------------------------
for i in range(len(bam_path)):
    cmd = f"python main.py seq2embd -H {hdf_path[i]} -E {embd_fold[i]} -b 100" 
    if not os.path.exists(embd_fold[i]): 
        print("run step2...")
        subprocess.run(cmd, shell=True)
    continue

## step 3.1 Feature Selection --------------------------
# output feature to temp/feature folder
if not skip_feature_selection:
    if os.path.exists(feature_fold): shutil.rmtree(feature_fold)
    for chromosome in Selector(feature_fold).chromosome_list:
        # do a set of samples together
        Selector(feature_fold).addFeature(embd_fold[:-1], chromosome=chromosome)
        # add one more sample 
        Selector(feature_fold).addFeature(embd_fold[-1], chromosome=chromosome)

## getFeature --------------------------
# feature selection step 2 resource consuming
Selector(feature_fold).getFeature()

## go back to feature selection, extract embedding -----------------
for chromosome in Selector(feature_fold).chromosome_list:
    Selector(feature_fold).applyFeature(embd_fold, chromosome)
## train PCA for deimension reduction
Selector(feature_fold).getIPCA(embd_fold)
## get each sample's representation
Selector(feature_fold).getRepre(embd_fold, repre_path)

## step 6. plot
# representation
repre = np.hstack([np.load(repre_path[i]) for i in range(len(bam_path))]).T
# for each column, if there are np.nan in that column, drop that column
repre = repre[:, np.all(~np.isnan(repre), axis=0)]  # comment for 0, un for 1
repre = np.nan_to_num(repre, nan=0.0)
# fit
repre_pca = sklearn.decomposition.PCA(n_components=2).fit_transform(repre)
# plot PCA 0 and 1

plt.scatter(repre_pca[:, 0], repre_pca[:, 1])
# plt.show()
# save figure 
plt.savefig("temp/plot.png")

print("Feature respresentation / finished..")
