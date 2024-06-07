import torch
import pandas as pd
import os
import sys
from src.embd2repre import Selector

torch.set_num_threads(os.cpu_count())

data_fold = "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data"

# task
task_id = int(sys.argv[1])
task = {
     3: {"chromosome": "10",},
     4: {"chromosome": "11",},
     5: {"chromosome": "13","hash_idx_start": 100000,},
     6: {"chromosome": "14",},
     7: {"chromosome": "15",},
     8: {"chromosome": "16",},
     9: {"chromosome": "17",},
    10: {"chromosome": "18",},
    11: {"chromosome": "20",},
    12: {"chromosome": "X",},
    13: {"chromosome": "1", "hash_idx_start": 220000,},
    14: {"chromosome": "2", "hash_idx_start":      0, "hash_idx_end":  50000,},
    15: {"chromosome": "2", "hash_idx_start":  50000, "hash_idx_end": 100000,},
    16: {"chromosome": "2", "hash_idx_start": 100000, "hash_idx_end": 150000,},
    17: {"chromosome": "2", "hash_idx_start": 150000, "hash_idx_end": 200000,},
    18: {"chromosome": "2", "hash_idx_start": 200000, "hash_idx_end": 250000,},
    19: {"chromosome": "3", "hash_idx_start":      0, "hash_idx_end":  50000,},
    20: {"chromosome": "3", "hash_idx_start":  50000, "hash_idx_end": 100000,},
    21: {"chromosome": "3", "hash_idx_start": 100000, "hash_idx_end": 150000,},
    22: {"chromosome": "3", "hash_idx_start": 150000, "hash_idx_end": 200000,},
}

# sample to be added to the selector
split = pd.read_csv(os.path.join(data_fold, "tcgaskcm/split.csv"))
split = split[split["train"] == 1]
profile = pd.read_csv(os.path.join(data_fold, "tcgaskcm/profile.csv"))
profile = profile[profile["patient"].isin(split["patient"])]
embd_fold = profile["embd_fold"].to_list()

# selector
selector = Selector(os.path.join(data_fold, "feature"))
selector.addFeature(embd_fold, **task[task_id])
