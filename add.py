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
     0: {"chromosome": "10", "hash_idx_start":  74100, "hash_idx_end":  94100},
     1: {"chromosome": "10", "hash_idx_start":  94100, "hash_idx_end": 114100},
     2: {"chromosome": "10", "hash_idx_start": 114100},
     3: {"chromosome": "11", "hash_idx_start":  60100, "hash_idx_end":  80100},
     4: {"chromosome": "11", "hash_idx_start":  80100, "hash_idx_end": 100100},
     5: {"chromosome": "11", "hash_idx_start": 100100, "hash_idx_end": 120000},
     6: {"chromosome": "11", "hash_idx_start": 120000},
     7: {"chromosome": "14", "hash_idx_start":  94500},
     8: {"chromosome": "15", "hash_idx_start":  83200},
     9: {"chromosome": "16", "hash_idx_start":  54300, "hash_idx_end": 74300},
    10: {"chromosome": "16", "hash_idx_start":  74300},
    11: {"chromosome": "17", "hash_idx_start":  45100, "hash_idx_end":  65100},
    12: {"chromosome": "17", "hash_idx_start":  65100},
    13: {"chromosome": "20", "hash_idx_start":  52500},
    14: {"chromosome":  "X", "hash_idx_start": 117100, "hash_idx_end": 137100},
    15: {"chromosome":  "X", "hash_idx_start": 137100},
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
