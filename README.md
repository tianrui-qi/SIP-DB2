# SIP-DB2

## Environment

The code is tested with `Python=3.10`, `PyTorch=2.2`, and `CUDA=11.8`. We 
recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are 
in place. To create an conda environment:
```bash
# clone the repository
git clone git@github.com:tianrui-qi/SIP-DB2.git
cd SIP-DB2
# create the conda environment
conda env create -f environment-cpu.yml     # cpu env
conda env create -f environment-gpu.yml     # gpu env
conda activate SIP-DB2
# uninstall triton to solve env problem
pip uninstall triton
```

## Data Structure

```
/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/
├── snps.csv
├── label.csv
├── profile.csv         # samples profile of stanford and tcgaskcm dataset
├── fastq/              # unaligned reads
│   ├── SRR8924580/         # sample id
│   │   ├── *R1.fastq.gz        # read 1
│   │   └── *R2.fastq.gz        # read 2
│   └── ...
├── ref/                # reference for alignment
│   ├── ref.fa              # reference genome
│   └── ...                 # corresponding index
├── bwa-mem2-2.2.1_x64-linux
├── sam/                # aligned reads by bwa-mem2
│   ├── SRR8924580.sam
│   └── ...
├── bam/                # sorted aligned reads by Samtools
│   ├── SRR8924580.bam      # sorted bam file
│   ├── SRR8924580.bam.bai  # corresponding index
│   └── ...
├── embd/
│   ├── SRR8924580/
│   │   ├── 1/                  # chr 1 embedding
│   │   │   ├── 000/
│   │   │   │   ├── 000.npy
│   │   │   │   └── ...
│   │   │   ├── ...
│   │   │   └── feature.npy         # chr 1 embedding after feature selection
│   │   ├── ...
│   │   ├── X/                  # chr X embedding
│   │   └── sequence.h5         # sequence of sample
│   ├── ...
│   ├── TCGA-3N-A9WB-06A/
│   └── ...
```
