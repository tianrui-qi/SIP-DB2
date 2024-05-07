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
