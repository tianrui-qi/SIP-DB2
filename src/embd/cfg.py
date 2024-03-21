__all__ = []


class Config:
    def __init__(self, **kwargs) -> None:
        data_fold = "data/stanford/"
        self.trainset = {
            "num": 128 * 8 * 1000,
            # path
            "snp_load_path": data_fold + "snps.tsv",
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
        }
        self.validset = {
            "num": 128 * 8 * 200,
            # path
            "snp_load_path": data_fold + "snps.tsv",
            "bam_load_fold": data_fold + "bam/",
            # sample
            "sample_list": [    # list of sample (id, label)
                ("SRR8924597", 0),  # normal skin   pre  anti-PD-1
                ("SRR8924596", 1),  # BCC tumor     post anti-PD-1
            ],
            # filter
            "pval_threshold": 1e-3,
            "pos_range": 64,    # since averge read length is 124
        }
        self.model = {
            "feats_token": [768, 4096, 4096, 768],
            "feats_coord": [2, 1024, 768],
            "feats_final": [768, 1],
        }
        self.runner = {
            # train
            "device": "cuda",
            "max_epoch": 800, 
            "accumu_steps": 8, 
            # checkpoint
            "ckpt_save_fold": "ckpt/embd-finetune/",
            "ckpt_load_path": "",
            "ckpt_load_lr"  : False,    # load lr from ckpt
            # data
            "batch_size": 128, 
            "num_workers": 8,
            # optimizer
            "lr": 5e-4,
        }
