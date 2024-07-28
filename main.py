import numpy as np
import pandas as pd
import sklearn.decomposition

import os
import json
import argparse
import subprocess
import matplotlib
import matplotlib.pyplot as plt

import src


matplotlib.use('Agg')


def main() -> None:
    # step                  ~ level verbal                  ~ level parallel
    # step1 bam2seq         sample * [chromosome, read]     sample
    # step2 seq2embd        sample * [chromosome, batch]    sample 
    # step3 addFeature      region * [hash]                 region, max 316
    # step4 getFeature      chromosome * [hash]             chromosome 
    # step5 applyFeature    chromosome * [sample, hash]     sample 
    # step6 getIPCA         [sample]                        no 
    # step7 getRepre        [sample]                        sample 
    # step8 downstream      [sample]                        no

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", type=str, required=True, dest="mode", 
        choices=["instance", "slurm"] + [f"step{i}" for i in range(1, 9)]
    )
    parser.add_argument("-c", type=str, required=True, dest="config_path")
    parser.add_argument("--task_id" , type=int, dest="task_id" , default=0)
    parser.add_argument("--task_num", type=int, dest="task_num", default=1)
    args = parser.parse_args()

    # configuration
    with open(args.config_path) as f: config = json.load(f)
    args_path  = config["args_path" ]
    args_step  = config["args_step" ]
    args_task  = {"task_id": args.task_id, "task_num": args.task_num}
    args_slurm = config["args_slurm"]
    
    # if  mode is instance or step{i}
    if args.mode in ["instance"] + [f"step{i}" for i in range(1, 9)]:
        # get list of bam_path and embd_fold for each sample
        args_path["bam_path"] = pd.read_csv(
            args_path["bam_path"], header=None,
        )[0].tolist()
        args_path["embd_fold"] = [
            os.path.join(
                args_path["embd_fold"],
                os.path.basename(b).replace('.bam', '')
            ) for b in args_path["bam_path"]
        ]
        # if mode is instance or step{i}
        func_list = [step1, step2, step3, step4, step5, step6, step7, step8]
        for i in range(1, 9):
            if args.mode not in ["instance", f"step{i}"]: continue
            func_list[i-1](**args_path, **args_step[f"step{i}"], **args_task)

    # if mode is slurm
    if args.mode == "slurm":
        dependency = None
        for i in range(1, 9):
            job_id = slurm(
                job_name=f"step{i}", dependency=dependency,
                **args_path, **args_slurm[f"step{i}"],
            )
            print((
                f"job (name: step{i}, id: {job_id}) submitted " 
                f"with dependency {dependency}"
            ))
            dependency = job_id


def step1(  # bam2seq (BAM and SNPs to Sequence)
    embd_fold: list[str], bam_path: list[str], snps_path: str, 
    quality_thresh: int = 16, length_thresh: int = 96, 
    verbal: bool | int = True,
    task_id: int = 0, task_num: int = 1, *vargs, **kwargs
) -> None:
    for i in range(task_id, len(embd_fold), task_num):
        src.bam2seq(
            bam_path[i], snps_path, os.path.join(embd_fold[i], "sequence.h5"), 
            quality_thresh=quality_thresh, length_thresh=length_thresh,
            verbal=verbal
        )


def step2(  # seq2embd (embedding calculation using DNABERT2)
    embd_fold: list[str], 
    pval_thresh: float = 0, batch_size: int = 100, verbal: bool | int = True,
    task_id: int = 0, task_num: int = 1, *vargs, **kwargs
) -> None:
    for i in range(task_id, len(embd_fold), task_num):
        if any(
            os.path.isdir(os.path.join(embd_fold[i], e)) 
            for e in os.listdir(embd_fold[i])
        ): continue
        src.seq2embd(
            os.path.join(embd_fold[i], "sequence.h5"), embd_fold[i], 
            pval_thresh=pval_thresh, batch_size=batch_size, verbal=verbal
        )


def step3(  # addFeature (feature selection)
    embd_fold: list[str], feature_fold: str, verbal: bool | int = True,
    task_id: int = 0, task_num: int = 1, *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)

    # max task_num is 316 since hash_step is 10000
    # decrease hash_step will increase task_num
    hash_step = 10000
    region = [
        {"chromosome": c, "hash_idx_start": i, "hash_idx_end": i+hash_step} 
        for c in selector.chromosome_list 
        for i in range(0, selector.hash_idx_max[c], hash_step)
    ]

    for i in range(task_id, len(region), task_num):
        selector.addFeature(embd_fold, **region[i], verbal=verbal)


def step4(  # getFeature
    feature_fold: str, verbal: bool | int = True,
    task_id: int = 0, task_num: int = 1, *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    for chromosome in selector.chromosome_list[task_id::task_num]:
        selector.getFeature(chromosome, verbal=verbal)


def step5(  # applyFeature (go back to feature selection, extract embedding)
    embd_fold: list[str], feature_fold: str, 
    top_k: float = 0.15, verbal: bool | int = True,
    task_id: int = 0, task_num: int = 1, *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    for chromosome in selector.chromosome_list:
        selector.applyFeature(
            embd_fold[task_id::task_num], chromosome, top_k=top_k, verbal=verbal
        )


def step6(  # getIPCA (train PCA for deimension reduction)
    embd_fold: list[str], feature_fold: str, 
    batch_size: int = 1, verbal: bool | int = True, *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    selector.getIPCA(embd_fold, batch_size, verbal=verbal)


def step7(  # getRepre (get each sample's representation)
    embd_fold: list[str], feature_fold: str, 
    recalculate: bool = False, verbal: bool | int = True,
    task_id: int = 0, task_num: int = 1, *vargs, **kwargs
) -> None:
    selector = src.Selector(feature_fold)
    selector.getRepre(
        embd_fold[task_id::task_num], recalculate=recalculate, verbal=verbal
    )


def step8(  # downstream analysis
    embd_fold: list[str], feature_fold: str, figure_path: str, 
    verbal: bool | int = True, *vargs, **kwargs
) -> None:
    # representation
    selector = src.Selector(feature_fold)
    repre = selector.getRepre(embd_fold, recalculate=False, verbal=verbal)
    # fit
    repre_pca = sklearn.decomposition.PCA(n_components=2).fit_transform(repre)
    # plot PCA 0 and 1
    plt.scatter(repre_pca[:, 0], repre_pca[:, 1])
    plt.savefig(figure_path)


def slurm(
    job_name: str, logs_fold: str, cpu: int, mem: int, gpu: int,
    environment: str, config_path: str, task_num: int = 1, 
    dependency: str | list[str] = None, *vargs, **kwargs
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
        f"{environment}" "\n"
        f"srun python main.py -m {job_name} -c {config_path} "
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