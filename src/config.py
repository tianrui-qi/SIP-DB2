class Config:
    def __init__(self) -> None:
        self.Trainer = {
            # train
            "max_epoch": 800, 
            "accumu_steps": 256, 
            # checkpoint
            "ckpt_save_fold": "", 
            "ckpt_load_path": "", 
            "ckpt_load_lr"  : "",
            # data
            "batch_size": 1, 
            # optimizer
            "lr": 1e-4,
        }
        self.FCN = {
            "feats": [768, 4096, 4096, 4096, 4096, 1]
        }
        self.ResAttNet = {
            "feats": [256, 512, 1024, 2045, 4096, 1],
            "use_cbam": False,
            "use_res" : False,
        }
