import pandas as pd
import os
import sys
from src.embd2repre import Selector

# task
task_id = int(sys.argv[1])
task = {
     1: {"chromosome":  "1", "start":   0, "end": 120},  2: {"chromosome":  "1", "start": 120, "end": 240},
     3: {"chromosome":  "1", "start": 240, "end": 360},  4: {"chromosome":  "1", "start": 360, "end": 480},
     5: {"chromosome":  "1", "start": 480, "end": 600},  6: {"chromosome":  "1", "start": 600, "end": 720},
     7: {"chromosome":  "1", "start": 720, "end": 840},  8: {"chromosome":  "1", "start": 840, "end": 960},
     9: {"chromosome":  "2", "start":   0, "end": 120}, 10: {"chromosome":  "2", "start": 120, "end": 240},
    11: {"chromosome":  "2", "start": 240, "end": 360}, 12: {"chromosome":  "2", "start": 360, "end": 480},
    13: {"chromosome":  "2", "start": 480, "end": 600}, 14: {"chromosome":  "2", "start": 600, "end": 720},
    15: {"chromosome":  "2", "start": 720, "end": 840}, 16: {"chromosome":  "2", "start": 840, "end": 960},
    17: {"chromosome":  "3", "start":   0, "end": 120}, 18: {"chromosome":  "3", "start": 120, "end": 240},
    19: {"chromosome":  "3", "start": 240, "end": 360}, 20: {"chromosome":  "3", "start": 360, "end": 480},
    21: {"chromosome":  "3", "start": 480, "end": 600}, 22: {"chromosome":  "3", "start": 600, "end": 720},
    23: {"chromosome":  "3", "start": 720, "end": 840}, 24: {"chromosome":  "3", "start": 840, "end": 960},
    25: {"chromosome":  "4", "start":   0, "end": 120}, 26: {"chromosome":  "4", "start": 120, "end": 240},
    27: {"chromosome":  "4", "start": 240, "end": 360}, 28: {"chromosome":  "4", "start": 360, "end": 480},
    29: {"chromosome":  "4", "start": 480, "end": 600}, 30: {"chromosome":  "4", "start": 600, "end": 720},
    31: {"chromosome":  "4", "start": 720, "end": 840}, 32: {"chromosome":  "4", "start": 840, "end": 960},
}

# sample
data_fold = "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data"
profile = pd.read_csv(os.path.join(data_fold, "stanford/profile.csv"))
embd_fold  = profile["embd_fold"].to_list()
profile = pd.read_csv(os.path.join(data_fold, "tcgaskcm/profile.csv"))
embd_fold += profile["embd_fold"].dropna().to_list()

# apply
selector = Selector(os.path.join(data_fold, "feature"))
selector.applyFeature(
    embd_fold=embd_fold[task[task_id]["start"]:task[task_id]["end"]], 
    chromosome=task[task_id]["chromosome"]
)
