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
    # 1
    0:  {"chromosome": "1", "hash_idx_start":      0, "hash_idx_end":  40000},
    1:  {"chromosome": "1", "hash_idx_start":  40000, "hash_idx_end":  80000},
    2:  {"chromosome": "1", "hash_idx_start":  80000, "hash_idx_end": 120000},
    3:  {"chromosome": "1", "hash_idx_start": 120000, "hash_idx_end": 160000},
    4:  {"chromosome": "1", "hash_idx_start": 160000, "hash_idx_end": 200000},
    5:  {"chromosome": "1", "hash_idx_start": 200000, "hash_idx_end": 240000},
    6:  {"chromosome": "1", "hash_idx_start": 240000, "hash_idx_end": 280000},
    # 2: 243000
    7:  {"chromosome": "2", "hash_idx_start":      0, "hash_idx_end":  40000},
    8:  {"chromosome": "2", "hash_idx_start":  40000, "hash_idx_end":  80000},
    9:  {"chromosome": "2", "hash_idx_start":  80000, "hash_idx_end": 120000},
    10: {"chromosome": "2", "hash_idx_start": 120000, "hash_idx_end": 160000},
    11: {"chromosome": "2", "hash_idx_start": 160000, "hash_idx_end": 200000},
    12: {"chromosome": "2", "hash_idx_start": 240000, "hash_idx_end": 280000},
    # 3: 199000
    13: {"chromosome": "3", "hash_idx_start":      0, "hash_idx_end":  40000},
    14: {"chromosome": "3", "hash_idx_start":  40000, "hash_idx_end":  80000},
    15: {"chromosome": "3", "hash_idx_start":  80000, "hash_idx_end": 120000},
    16: {"chromosome": "3", "hash_idx_start": 120000, "hash_idx_end": 160000},
    17: {"chromosome": "3", "hash_idx_start": 160000, "hash_idx_end": 200000},
    # 4: 191000
    18: {"chromosome": "4", "hash_idx_start":      0, "hash_idx_end":  40000},
    19: {"chromosome": "4", "hash_idx_start":  40000, "hash_idx_end":  80000},
    20: {"chromosome": "4", "hash_idx_start":  80000, "hash_idx_end": 120000},
    21: {"chromosome": "4", "hash_idx_start": 120000, "hash_idx_end": 160000},
    22: {"chromosome": "4", "hash_idx_start": 160000, "hash_idx_end": 200000},
    # 5: 182000
    23: {"chromosome": "5", "hash_idx_start":      0, "hash_idx_end":  40000},
    24: {"chromosome": "5", "hash_idx_start":  40000, "hash_idx_end":  80000},
    25: {"chromosome": "5", "hash_idx_start":  80000, "hash_idx_end": 120000},
    26: {"chromosome": "5", "hash_idx_start": 120000, "hash_idx_end": 160000},
    27: {"chromosome": "5", "hash_idx_start": 160000, "hash_idx_end": 200000},
    # 6: 171000
    28: {"chromosome": "6", "hash_idx_start":      0, "hash_idx_end":  40000},
    29: {"chromosome": "6", "hash_idx_start":  40000, "hash_idx_end":  80000},
    30: {"chromosome": "6", "hash_idx_start":  80000, "hash_idx_end": 120000},
    31: {"chromosome": "6", "hash_idx_start": 120000, "hash_idx_end": 160000},
    32: {"chromosome": "6", "hash_idx_start": 160000, "hash_idx_end": 200000},
    # 7: 160000
    33: {"chromosome": "7", "hash_idx_start":      0, "hash_idx_end":  40000},
    34: {"chromosome": "7", "hash_idx_start":  40000, "hash_idx_end":  80000},
    35: {"chromosome": "7", "hash_idx_start":  80000, "hash_idx_end": 120000},
    36: {"chromosome": "7", "hash_idx_start": 120000, "hash_idx_end": 160000},
    # 8: 146000
    37: {"chromosome": "8", "hash_idx_start":      0, "hash_idx_end":  40000},
    38: {"chromosome": "8", "hash_idx_start":  40000, "hash_idx_end":  80000},
    39: {"chromosome": "8", "hash_idx_start":  80000, "hash_idx_end": 120000},
    40: {"chromosome": "8", "hash_idx_start": 120000, "hash_idx_end": 160000},
    # 9: 139000
    41: {"chromosome": "9", "hash_idx_start":      0, "hash_idx_end":  40000},
    42: {"chromosome": "9", "hash_idx_start":  40000, "hash_idx_end":  80000},
    43: {"chromosome": "9", "hash_idx_start":  80000, "hash_idx_end": 120000},
    44: {"chromosome": "9", "hash_idx_start": 120000, "hash_idx_end": 160000},
    # 10: 134000
    45: {"chromosome": "10", "hash_idx_start":      0, "hash_idx_end":  40000},
    46: {"chromosome": "10", "hash_idx_start":  40000, "hash_idx_end":  80000},
    47: {"chromosome": "10", "hash_idx_start":  80000, "hash_idx_end": 120000},
    48: {"chromosome": "10", "hash_idx_start": 120000, "hash_idx_end": 160000},
    # 11: 136000
    49: {"chromosome": "11", "hash_idx_start":      0, "hash_idx_end":  40000},
    50: {"chromosome": "11", "hash_idx_start":  40000, "hash_idx_end":  80000},
    51: {"chromosome": "11", "hash_idx_start":  80000, "hash_idx_end": 120000},
    52: {"chromosome": "11", "hash_idx_start": 120000, "hash_idx_end": 160000},
    # 12: 134000
    53: {"chromosome": "12", "hash_idx_start":      0, "hash_idx_end":  40000},
    54: {"chromosome": "12", "hash_idx_start":  40000, "hash_idx_end":  80000},
    55: {"chromosome": "12", "hash_idx_start":  80000, "hash_idx_end": 120000},
    56: {"chromosome": "12", "hash_idx_start": 120000, "hash_idx_end": 160000},
    # 13: 115000
    57: {"chromosome": "13", "hash_idx_start":      0, "hash_idx_end":  40000},
    58: {"chromosome": "13", "hash_idx_start":  40000, "hash_idx_end":  80000},
    59: {"chromosome": "13", "hash_idx_start":  80000, "hash_idx_end": 120000},
    # 14: 108000
    60: {"chromosome": "14", "hash_idx_start":      0, "hash_idx_end":  40000},
    61: {"chromosome": "14", "hash_idx_start":  40000, "hash_idx_end":  80000},
    62: {"chromosome": "14", "hash_idx_start":  80000, "hash_idx_end": 120000},
    # 15: 102000
    63: {"chromosome": "15", "hash_idx_start":      0, "hash_idx_end":  40000},
    64: {"chromosome": "15", "hash_idx_start":  40000, "hash_idx_end":  80000},
    65: {"chromosome": "15", "hash_idx_start":  80000, "hash_idx_end": 120000},
    # 16:  91000
    66: {"chromosome": "16", "hash_idx_start":      0, "hash_idx_end":  40000},
    67: {"chromosome": "16", "hash_idx_start":  40000, "hash_idx_end":  80000},
    68: {"chromosome": "16", "hash_idx_start":  80000, "hash_idx_end": 120000},
    # 17:  84000
    69: {"chromosome": "17", "hash_idx_start":      0, "hash_idx_end":  40000},
    70: {"chromosome": "17", "hash_idx_start":  40000, "hash_idx_end":  80000},
    71: {"chromosome": "17", "hash_idx_start":  80000, "hash_idx_end": 120000},
    # 18:  81000
    72: {"chromosome": "18", "hash_idx_start":      0, "hash_idx_end":  40000},
    73: {"chromosome": "18", "hash_idx_start":  40000, "hash_idx_end":  80000},
    74: {"chromosome": "18", "hash_idx_start":  80000, "hash_idx_end": 120000},
    # 19:  59000
    75: {"chromosome": "19", "hash_idx_start":      0, "hash_idx_end":  40000},
    76: {"chromosome": "19", "hash_idx_start":  40000, "hash_idx_end":  80000},
    # 20:  65000
    77: {"chromosome": "20", "hash_idx_start":      0, "hash_idx_end":  40000},
    78: {"chromosome": "20", "hash_idx_start":  40000, "hash_idx_end":  80000},
    # 21:  47000
    79: {"chromosome": "21", "hash_idx_start":      0, "hash_idx_end":  40000},
    80: {"chromosome": "21", "hash_idx_start":  40000, "hash_idx_end":  80000},
    # 22:  51000
    81: {"chromosome": "22", "hash_idx_start":      0, "hash_idx_end":  40000},
    82: {"chromosome": "22", "hash_idx_start":  40000, "hash_idx_end":  80000},
    # X:  157000
    83: {"chromosome": "X", "hash_idx_start":      0, "hash_idx_end":  40000},
    84: {"chromosome": "X", "hash_idx_start":  40000, "hash_idx_end":  80000},
    85: {"chromosome": "X", "hash_idx_start":  80000, "hash_idx_end": 120000},
    86: {"chromosome": "X", "hash_idx_start": 120000, "hash_idx_end": 160000},
}

# sample to be added to the selector
profile = pd.read_csv(os.path.join(data_fold, "profile.csv"))
profile = profile[profile["train"]==1]
embd_fold = profile["embd_fold"].to_list()

# selector
selector = Selector(os.path.join(data_fold, "feature"))
selector.addFeature(embd_fold, **task[task_id])
