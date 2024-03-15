# TCGA-Onc

For more details, please refer to 
[PowerPoint](https://regn-my.sharepoint.com/:p:/r/personal/jing_he1_regeneron_com/_layouts/15/Doc.aspx?sourcedoc=%7B6811AE49-3195-42AB-BFEA-DAFC7561C154%7D&file=20240102_cancer_mol_pheno.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1&login_hint=tianrui.qi%40regeneron.com&ct=1710520631940&wdOrigin=OFFICECOM-WEB.MAIN.REC&cid=46ed813f-81a2-47cd-8bc3-85043f2b2341&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=5f170870-0192-43d8-9d99-c4dfbcd60346) 
and [Documentation](https://craft.tianrui-qi.com/tcga-onc).

## Environment

The code is tested with `Python=3.9`, `PyTorch=2.2`, and `CUDA=12.1`. We 
recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are 
in place. To create an conda environment:
```bash
# clone the repository
git clone git@github.com:tianrui-qi/TCGA-Onc.git
cd TCGA-Onc
# create the conda environment
conda env create -f environment-cpu.yml     # cpu env
conda env create -f environment-gpu.yml     # gpu env
conda activate TCGA-Onc
# uninstall triton to solve env problem
pip uninstall triton
```
