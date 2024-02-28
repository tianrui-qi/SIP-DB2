__all__ = ["Config"]


class Config:
    def __init__(self, **kwargs) -> None:
        self.trainset = {
            "len": 4096 * 100,
            "bam_load_fold": "data/bam",
            "sample_list": [    # list of sample (id, label)
                ("SRR8924593", 0),  # BCC tumor     pre  anti-PD-1
                ("SRR8924590", 1),  # BCC tumor     post anti-PD-1
                ("SRR8924591", 0),  # normal skin   pre anti-PD-1
                ("SRR8924592", 1),  # normal skin   post anti-PD-1
            ],
            "quality_threshold": 16,
            "length_threshold" : 96,
        }
        self.model = {
            "feats_coord": [2, 2048, 768],
            "feats": [768, 8192, 8192, 4096, 4096, 1]
        }
        self.runner = {
            # train
            "max_epoch": 800, 
            "accumu_steps": 1, 
            # checkpoint
            "ckpt_save_fold": "ckpt/" + self.__class__.__name__,
            "ckpt_load_path": "",       # path without .ckpt
            "ckpt_load_lr"  : False,    # load lr from ckpt
            # data
            "batch_size": 4096, 
            "num_workers": 38,
            # optimizer
            "lr": 1e-4,
        }
