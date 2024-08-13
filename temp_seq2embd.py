import torch
import transformers

import numpy as np
import pandas as pd

import os
import sys
import tqdm
import warnings

from src.embd.model import PretrainModel


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton*")


def seq2embd(
    hdf_load_path: str, embd_save_fold: str, chromosome: str, 
    pval_thresh: float = 1e-4, batch_size: int = 1000,
    verbal: bool | int = True, *vargs, **kwargs
) -> None:
    if os.path.exists(os.path.join(embd_save_fold, f"{chromosome}.npy")):
        print(f"skip:seq2embd:calculate:{embd_save_fold}:{chromosome}")
        return

    pval_thresh_list = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", trust_remote_code=True
    )
    model = PretrainModel()
    model.eval().to(device)

    # get the hdf that store sequence and p-value for filtering
    hdf = pd.read_hdf(hdf_load_path, key=f"/chr{chromosome}", mode="r")
    if pval_thresh != 0: hdf = hdf[hdf[f"{pval_thresh:.0e}"]>=1]

    embd = None
    for i in tqdm.tqdm(
        range(int(np.ceil(len(hdf)/batch_size))), unit="batch", 
        desc=f"seq2embd:calculate:{embd_save_fold}:{chromosome}", 
        smoothing=0.0, dynamic_ncols=True, 
        disable=(not verbal) if isinstance(verbal, bool) else (0 > verbal),
    ):
        # sequence
        sequence_batch = hdf["sequence"].iloc[
            i*batch_size:(i+1)*batch_size
        ].to_list()
        # token
        token_batch = tokenizer(
            sequence_batch, return_tensors = 'pt', padding=True
        )["input_ids"].to(device)
        # embedding
        with torch.no_grad():
            embedding_batch = model(token_batch).detach().cpu().numpy()
        # save
        embd = np.concatenate(
            [embd, embedding_batch], axis=0
        ) if embd is not None else embedding_batch
    embd: np.ndarray = np.concatenate([
        # [0:768] for embedding
        embd,
        # [768] for position                       
        hdf["pos"].to_numpy().reshape(-1, 1),
        # [769:776] for # of variants with p-value <= 1e-[0:7] cover by read
        np.concatenate([
            hdf[f"{_:.0e}"].to_numpy().reshape(-1, 1) for _ in pval_thresh_list
        ], axis=1, dtype=np.float32)
    ], axis=1, dtype=np.float32)
    embd = embd[embd[:, 768].argsort()]

    if not os.path.exists(embd_save_fold): os.makedirs(embd_save_fold)
    np.save(os.path.join(embd_save_fold, f"{chromosome}.npy"), embd)


if __name__ == "__main__":
    sample = [
        'SRR8924591', 'SRR8924590', 'SRR8924593', 'SRR8924592', 'SRR8924594', 
        'SRR8924595', 'SRR8924597', 'SRR8924596', 'SRR8924599', 'SRR8924598', 
        'SRR8924585', 'SRR8924584', 'SRR8924582', 'SRR8924583', 'SRR8924589', 
        'SRR8924587', 'SRR8924588', 'SRR8924586', 'SRR8924580', 'SRR8924581', 
        'SRR8924601', 'SRR8924600', 'SRR8924602', 
        'TCGA-D3-A3ML-06A', 'TCGA-D3-A3ML-10A', 'TCGA-D3-A8GP-06A', 
        'TCGA-D3-A8GP-10A', 'TCGA-D9-A6EC-06A', 'TCGA-D9-A6EC-10A', 
        'TCGA-EB-A5VV-06A', 'TCGA-EB-A5VV-10A', 'TCGA-EE-A2GK-06A', 
        'TCGA-EE-A2GK-10A', 'TCGA-EE-A2ME-06A', 'TCGA-EE-A2ME-10A', 
        'TCGA-EE-A3AD-06A', 'TCGA-EE-A3AD-10A', 'TCGA-EE-A3AH-06A', 
        'TCGA-EE-A3AH-10A', 'TCGA-EE-A3JD-06A', 'TCGA-EE-A3JD-10A', 
        'TCGA-FR-A7UA-06A', 'TCGA-FR-A7UA-10A', 'TCGA-FS-A1ZY-06A', 
        'TCGA-FS-A1ZY-10A', 'TCGA-FW-A5DY-06A', 'TCGA-FW-A5DY-11A'
    ]
    embd_fold = [   # store hdf file to be loaded
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924591', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924590', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924593', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924592', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924594', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924595', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924597', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924596', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924599', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924598', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924585', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924584', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924582', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924583', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924589', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924587', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924588', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924586', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924580', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924581', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924601', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924600', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/SRR8924602', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-D3-A3ML-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-D3-A3ML-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-D3-A8GP-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-D3-A8GP-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-D9-A6EC-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-D9-A6EC-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EB-A5VV-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EB-A5VV-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A2GK-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A2GK-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A2ME-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A2ME-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A3AD-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A3AD-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A3AH-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A3AH-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A3JD-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-EE-A3JD-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-FR-A7UA-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-FR-A7UA-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-FS-A1ZY-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-FS-A1ZY-10A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-FW-A5DY-06A', 
        '/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/data/develop/v2/embd/TCGA-FW-A5DY-11A'
    ]
    hdf_path = [os.path.join(embd_fold[i], "sequence.h5") for i in range(len(sample))]
    embd_fold = [os.path.join(
        "/mnt/efs_v2/dbgap_tcga/users/tianrui.qi/SIP-DB2/develop/v1/embd", sample[i]
    ) for i in range(len(sample))]

    task_num = 40
    task_id = int(sys.argv[1])
    task = []
    for i in range(len(sample)):
        for c in [str(i) for i in range(1, 23)] + ["X"]:
            pval_thresh = 1e-4 if sample[i].startswith("SRR") else 1e-3
            task.append((hdf_path[i], embd_fold[i], c, pval_thresh))

    for i in range(task_id, len(task), task_num): seq2embd(*task[i])
