# SIP-DB2

Documentation of the data processing pipeline is available at 
[Craft](https://craft.tianrui-qi.com/sip-db2).

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

## Acknowledgement

I would like to express my sincere gratitude to my manager, 
Dr. [Jing He](https://www.linkedin.com/in/jing-he-0ba21622/), for providing this
opportunity to co-op at 
[Regeneron Genetics Center](https://www.regeneron.com/science/genetics-center) 
and for her invaluable guidance on both the project and personal development 
throughout my co-op experience. 
I also had a great time working with oncology group, 
Dr. [Silvia Alvarez](https://www.linkedin.com/in/silvia-alvarez-phd-b32a7628/),
Dr. [Jessie Brown](https://www.linkedin.com/in/jessie-brown-22151b85/),
Dr. [Adolfo Ferrando](https://www.linkedin.com/in/adolfo-ferrando-7a79b35/), and
Dr. [Hossein Khiabanian](https://www.linkedin.com/in/hosseink/).
Special thanks to, not only colleagues but also close friends,
[Jie Peng](https://www.linkedin.com/in/jie-peng-600178253/), 
[Zhenyu Zhang](https://www.linkedin.com/in/zhenyu-zhang-825a09251/), and 
[Shiying Zheng](https://www.linkedin.com/in/shiying-zheng00/), whose 
encouragement helped me navigate challenging times and made my after-work life 
enriching and enjoyable.
