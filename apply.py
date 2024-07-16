import pandas as pd
import os
import sys
from src.embd2repre import Selector

data_fold = "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data"

# task
task_id = int(sys.argv[1])
task = {
    0:  {"s": 0, "e": 330, "c":  "1"}, 1:  {"s": 330, "e": 660, "c":  "1"}, 2:  {"s": 660, "e": 967, "c":  "1"},
    3:  {"s": 0, "e": 330, "c":  "2"}, 4:  {"s": 330, "e": 660, "c":  "2"}, 5:  {"s": 660, "e": 967, "c":  "2"},
    6:  {"s": 0, "e": 330, "c":  "3"}, 7:  {"s": 330, "e": 660, "c":  "3"}, 8:  {"s": 660, "e": 967, "c":  "3"},
    9:  {"s": 0, "e": 330, "c":  "4"}, 10: {"s": 330, "e": 660, "c":  "4"}, 11: {"s": 660, "e": 967, "c":  "4"},
    12: {"s": 0, "e": 330, "c":  "5"}, 13: {"s": 330, "e": 660, "c":  "5"}, 14: {"s": 660, "e": 967, "c":  "5"},
    15: {"s": 0, "e": 330, "c":  "6"}, 16: {"s": 330, "e": 660, "c":  "6"}, 17: {"s": 660, "e": 967, "c":  "6"},
    18: {"s": 0, "e": 330, "c":  "7"}, 19: {"s": 330, "e": 660, "c":  "7"}, 20: {"s": 660, "e": 967, "c":  "7"},
    21: {"s": 0, "e": 330, "c":  "8"}, 22: {"s": 330, "e": 660, "c":  "8"}, 23: {"s": 660, "e": 967, "c":  "8"},
    24: {"s": 0, "e": 330, "c":  "9"}, 25: {"s": 330, "e": 660, "c":  "9"}, 26: {"s": 660, "e": 967, "c":  "9"},
    27: {"s": 0, "e": 330, "c": "10"}, 28: {"s": 330, "e": 660, "c": "10"}, 29: {"s": 660, "e": 967, "c": "10"},
    30: {"s": 0, "e": 330, "c": "11"}, 31: {"s": 330, "e": 660, "c": "11"}, 32: {"s": 660, "e": 967, "c": "11"},
    33: {"s": 0, "e": 330, "c": "12"}, 34: {"s": 330, "e": 660, "c": "12"}, 35: {"s": 660, "e": 967, "c": "12"},
    36: {"s": 0, "e": 330, "c": "13"}, 37: {"s": 330, "e": 660, "c": "13"}, 38: {"s": 660, "e": 967, "c": "13"},
    39: {"s": 0, "e": 330, "c": "14"}, 40: {"s": 330, "e": 660, "c": "14"}, 41: {"s": 660, "e": 967, "c": "14"},
    42: {"s": 0, "e": 330, "c": "15"}, 43: {"s": 330, "e": 660, "c": "15"}, 44: {"s": 660, "e": 967, "c": "15"},
    45: {"s": 0, "e": 330, "c": "16"}, 46: {"s": 330, "e": 660, "c": "16"}, 47: {"s": 660, "e": 967, "c": "16"},
    48: {"s": 0, "e": 330, "c": "17"}, 49: {"s": 330, "e": 660, "c": "17"}, 50: {"s": 660, "e": 967, "c": "17"},
    51: {"s": 0, "e": 330, "c": "18"}, 52: {"s": 330, "e": 660, "c": "18"}, 53: {"s": 660, "e": 967, "c": "18"},
    54: {"s": 0, "e": 330, "c": "19"}, 55: {"s": 330, "e": 660, "c": "19"}, 56: {"s": 660, "e": 967, "c": "19"},
    57: {"s": 0, "e": 330, "c": "20"}, 58: {"s": 330, "e": 660, "c": "20"}, 59: {"s": 660, "e": 967, "c": "20"},
    60: {"s": 0, "e": 330, "c": "21"}, 61: {"s": 330, "e": 660, "c": "21"}, 62: {"s": 660, "e": 967, "c": "21"},
    63: {"s": 0, "e": 330, "c": "22"}, 64: {"s": 330, "e": 660, "c": "22"}, 65: {"s": 660, "e": 967, "c": "22"},
    66: {"s": 0, "e": 330, "c":  "X"}, 67: {"s": 330, "e": 660, "c":  "X"}, 68: {"s": 660, "e": 967, "c":  "X"},
}
s, e, c = task[task_id]["s"], task[task_id]["e"], task[task_id]["c"]

# sample
profile = pd.read_csv(os.path.join(data_fold, "profile.csv"))
embd_fold = profile["embd_fold"].to_list()

# selector
selector = Selector(os.path.join(data_fold, "feature"))
selector.applyFeature(embd_fold[s:e], c)
