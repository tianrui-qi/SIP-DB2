__all__ = ["Config"]


class Config:
    def __init__(self, **kwargs) -> None:
        data_fold = "/mnt/efs_v2/tag_onc/users/tianrui.qi/TCGA-Onc/data/"
        self.trainset = {
            "num": 64 * 16 * 1000,
            # path
            "snp_load_path": data_fold + "snp/snp_filter.tsv",
            "bam_load_fold": data_fold + "bam/",
            # sample
            "sample_list": [    # list of sample (id, label)
                ("SRR8924593", 0),  # BCC tumor     pre  anti-PD-1
                ("SRR8924590", 1),  # BCC tumor     post anti-PD-1
                ("SRR8924591", 0),  # normal skin   pre  anti-PD-1
                ("SRR8924592", 1),  # normal skin   post anti-PD-1
            ],
            # filter
            "pval_threshold": 1e-3,
            "pos_range": 64,    # since averge read length is 124
            "quality_threshold": 16,
            "length_threshold" : 96,
        }
        self.model = {
            "feats_coord": [2, 1024, 768],
            "feats_final": [768, 4096, 4096, 4096, 4096, 1]
        }
        self.runner = {
            # train
            "device": "cuda",
            "max_epoch": 800, 
            "accumu_steps": 16, 
            # checkpoint
            "ckpt_save_fold": "ckpt/dnabert2fc/",
            "ckpt_load_path": "",       # path without .ckpt
            "ckpt_load_lr"  : False,    # load lr from ckpt
            # data
            "batch_size": 64, 
            "num_workers": 8,
            # optimizer
            "lr": 5e-4,
            "T_max": 15,    # for CosineAnnealingLR
        }
