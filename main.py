import numpy as np
import sklearn.decomposition

import os
import pysam
import argparse
import subprocess
import matplotlib
import matplotlib.pyplot as plt

import src


matplotlib.use('Agg')


def main() -> None:
    # parameters
    data_fold = f"/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/temp/slurm"
    sample_name  = ["a", "b", "c", "d"]
    args_path = {
        "bam_path":     # list of bam path to be processed
            [os.path.join(data_fold, f"bam/{i}.bam") for i in sample_name], 
        "embd_fold":
            [os.path.join(data_fold, f"embd/{i}") for i in sample_name], 
        "feature_fold":
            os.path.join(data_fold, "feature"),
        "snps_path":    # step 1 only
            "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/snps.csv",
        "figure_path":  # step 8 only
            os.path.join(data_fold, "plot.png"),
        "logs_fold":    # slurm only
            os.path.join(data_fold, "logs"),
    }
    args_step = {
        "step0": {},
        "step1": {"quality_thresh": 16, "length_thresh": 96},
        "step2": {"pval_thresh": 0, "batch_size": 100},
        "step3": {},
        "step4": {},
        "step5": {},
        "step6": {"batch_size": 1},
        "step7": {},
        "step8": {}
    }
    args_slurm = {
        "step0":    # sample preparation: no parallel
            {"cpu":  2, "mem":  4, "gpu": 0, "task_num": 1},  
        "step1":    # bam2seq: sample level parallel
            {"cpu":  2, "mem": 32, "gpu": 0, "task_num": 2},
        "step2":    # seq2embd: sample level parallel
            {"cpu": 32, "mem": 32, "gpu": 0, "task_num": 2},
        "step3":    # addFeature: chromosome and region level parallel, max 316
            {"cpu": 12, "mem": 24, "gpu": 0, "task_num": 4},
        "step4":    # getFeature: chromosome level parallel
            {"cpu":  2, "mem":  8, "gpu": 0, "task_num": 4},
        "step5":    # applyFeature: sample level parallel
            {"cpu":  4, "mem": 10, "gpu": 0, "task_num": 2},
        "step6":    # getIPCA: no parallel
            {"cpu": 24, "mem": 64, "gpu": 0, "task_num": 1},
        "step7":    # getRepre: sample level parallel
            {"cpu":  2, "mem":  4, "gpu": 0, "task_num": 2},
        "step8":    # downstream analysis: no parallel
            {"cpu":  2, "mem":  4, "gpu": 0, "task_num": 1},
    }

    # input argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", type=str, required=True, dest="mode", 
        choices=["instance", "slurm"] + [f"step{i}" for i in range(9)]
    )
    parser.add_argument("--task_id" , type=int, dest="task_id" , default=0)
    parser.add_argument("--task_num", type=int, dest="task_num", default=1)
    args_task = vars(parser.parse_args())

    # if mode is instance or step{i}
    for i, step in enumerate(
        [step0, step1, step2, step3, step4, step5, step6, step7, step8]
    ):
        if args_task["mode"] not in ["instance", f"step{i}"]: continue
        step(**args_path, **args_step[f"step{i}"], **args_task)

    # if mode is slurm
    dependency = None
    for step in [f"step{i}" for i in range(9)]:
        if args_task["mode"] != "slurm": break
        dependency = slurm(
            job_name=step, dependency=dependency,
            **args_path, **args_slurm[step],
        )


def step0(  # sample preparation
    bam_path, *vargs, **kwargs
) -> None:
    def make_sample(bam_load_path, bam_save_path, num_reads=500):
        # file check
        os.makedirs(os.path.dirname(bam_save_path), exist_ok=True)
        if os.path.exists(bam_save_path): os.remove(bam_save_path)
        if os.path.exists(bam_save_path + ".bai"): 
            os.remove(bam_save_path + ".bai")
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
    data_fold = "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data"
    make_sample(os.path.join(data_fold, "bam/SRR8924580.bam"), bam_path[0])
    make_sample(os.path.join(data_fold, "bam/SRR8924581.bam"), bam_path[1])
    make_sample(os.path.join(data_fold, "bam/SRR8924582.bam"), bam_path[2])
    make_sample(os.path.join(data_fold, "bam/SRR8924583.bam"), bam_path[3])


def step1(  # bam2seq (BAM and SNPs to Sequence)
    embd_fold: list[str], bam_path: list[str], snps_path: str, 
    quality_thresh: int = 16, length_thresh: int = 96,
    task_id: int = 0, task_num: int = 1, 
    *vargs, **kwargs
) -> None:
    hdf_path = [
        os.path.join(embd_fold[i], "sequence.h5") for i in range(len(embd_fold))
    ]
    for i in range(task_id, len(bam_path), task_num):
        if os.path.exists(hdf_path[i]): continue
        src.bam2seq(
            bam_path[i], snps_path, hdf_path[i], 
            quality_thresh=quality_thresh, length_thresh=length_thresh
        )


def step2(  # seq2embd (embedding calculation using DNABERT2)
    embd_fold: list[str], 
    pval_thresh: float = 0, batch_size: int = 100,
    task_id: int = 0, task_num: int = 1, 
    *vargs, **kwargs
) -> None:
    hdf_path = [
        os.path.join(embd_fold[i], "sequence.h5") for i in range(len(embd_fold))
    ]
    for i in range(task_id, len(hdf_path), task_num):
        if any(
            os.path.isdir(os.path.join(embd_fold[i], d)) 
            for d in os.listdir(embd_fold[i])
        ): continue
        src.seq2embd(
            hdf_path[i], embd_fold[i], 
            pval_thresh=pval_thresh, batch_size=batch_size
        )


def step3(  # addFeature (feature selection)
    embd_fold: list[str], feature_fold: str, 
    task_id: int = 0, task_num: int = 1,
    *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)

    # max task_num is 316 since hash_step is 10000
    # decrease hash_step will increase task_num
    hash_step = 10000
    task = [
        {"chromosome": c, "hash_idx_start": i, "hash_idx_end": i+hash_step} 
        for c in selector.chromosome_list 
        for i in range(0, selector.hash_idx_max[c], hash_step)
    ]

    for i in range(task_id, len(task), task_num):
        selector.addFeature(embd_fold, **task[i])


def step4(  # getFeature
    feature_fold: str, 
    task_id: int = 0, task_num: int = 1,
    *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    for chromosome in selector.chromosome_list[task_id::task_num]:
        selector.getFeature(chromosome)


def step5(  # applyFeature (go back to feature selection, extract embedding)
    embd_fold: list[str], feature_fold: str, 
    task_id: int = 0, task_num: int = 1,
    *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    for chromosome in selector.chromosome_list:
        selector.applyFeature(embd_fold[task_id::task_num], chromosome)


def step6(  # getIPCA (train PCA for deimension reduction)
    embd_fold: list[str], feature_fold: str, 
    batch_size: int = 1,
    *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    selector.getIPCA(embd_fold, batch_size)


def step7(  # getRepre (get each sample's representation)
    embd_fold: list[str], feature_fold: str, 
    task_id: int = 0, task_num: int = 1,
    *vargs, **kwargs
) -> None:
    repre_path = [
        os.path.join(embd_fold[i], "representation.npy") 
        for i in range(len(embd_fold))
    ]
    selector = src.Selector(feature_fold)
    selector.getRepre(
        embd_fold[task_id::task_num], repre_path[task_id::task_num]
    )


def step8(  # downstream analysis
    embd_fold: list[str], figure_path: str, *vargs, **kwargs
) -> None:
    repre_path = [
        os.path.join(embd_fold[i], "representation.npy") 
        for i in range(len(embd_fold))
    ]
    # representation
    repre = np.hstack([
        np.load(repre_path[i]) for i in range(len(repre_path))
    ]).T
    # for each column, if there are np.nan in that column, drop that column
    repre = repre[:, np.all(~np.isnan(repre), axis=0)]  # comment for 0, un for 1
    repre = np.nan_to_num(repre, nan=0.0)
    # fit
    repre_pca = sklearn.decomposition.PCA(n_components=2).fit_transform(repre)
    # plot PCA 0 and 1
    plt.scatter(repre_pca[:, 0], repre_pca[:, 1])
    plt.savefig(figure_path)


def slurm(
    job_name: str, logs_fold: str, cpu: int, mem: int, gpu: int,
    task_num: int = 1, dependency: str | list[str] = None,
    *vargs, **kwargs
):
    # create sh
    sh_path = f"{job_name}.sh"
    with open(sh_path, 'w') as sh_file: sh_file.write((
        "#!/bin/bash" "\n"
        f"#SBATCH --job-name={job_name}" "\n"
        f"#SBATCH --output={logs_fold}/%A_{job_name}/%a.out" "\n"
        f"#SBATCH --error={logs_fold}/%A_{job_name}/%a.err" "\n"
        f"#SBATCH --cpus-per-task={cpu}" "\n"
        f"#SBATCH --mem={mem}G" "\n"
        f"#SBATCH --gres=gpu:{gpu}" "\n"
        f"#SBATCH --array=0-{task_num-1}" "\n"
        "#SBATCH --requeue" "\n"
        "source /home/s.tianrui.qi/miniconda3/etc/profile.d/conda.sh" "\n"
        "conda activate SIP-DB2" "\n"
        f"srun python main.py -m {job_name} "
        f"--task_id $SLURM_ARRAY_TASK_ID --task_num {task_num}" "\n"
    ))
    # submit job
    if isinstance(dependency, str): dependency = [dependency]
    if isinstance(dependency, list): dependency = ":".join(dependency)
    if dependency: result = subprocess.run(
        ["sbatch", f"--dependency=afterok:{dependency}", sh_path], 
        capture_output=True, text=True
    )
    else: result = subprocess.run(
        ["sbatch", sh_path], capture_output=True, text=True
    )
    job_id = result.stdout.strip().split()[-1]
    # remove sh
    os.remove(sh_path)
    return job_id


if __name__ == "__main__": main()