
import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

MODS = ["t1n", "t1c", "t2w", "t2f"]

def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data

def normalize_nonzero(x):
    mask = x > 0
    if mask.sum() == 0:
        return x
    v = x[mask]
    p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
    x = np.clip(x, p1, p99)
    x = (x - p1) / (p99 - p1 + 1e-8)
    return x

class GLIDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.patients = sorted([p for p in glob(os.path.join(root, "BraTS-GLI-*")) if os.path.isdir(p)])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pdir = self.patients[idx]
        pid = os.path.basename(pdir)

        vols = []
        for m in MODS:
            f = os.path.join(pdir, f"{pid}-{m}.nii.gz")
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing {m}: {f}")
            v = load_nii(f)
            v = normalize_nonzero(v)
            vols.append(v)

        x = np.stack(vols, axis=0)  # [4, H, W, D]
        x = torch.from_numpy(x)
        return x, pid
