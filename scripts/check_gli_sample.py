import os
from pathlib import Path

import nibabel as nib
import numpy as np

# ✅ Change this to your MRI folder root (where BraTS-GLI-* folders live)
# Example:
#   /home/craft-guest/sjaipurkar/projects/mri_realism/data/mri
DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", "./data/mri")).resolve()

# ✅ Put your sample patient ID here (folder name)
PATIENT_ID = os.environ.get("PATIENT_ID", "BraTS-GLI-00002-000")

# Common BraTS GLI modalities
MODS = ["t1n", "t1c", "t2w", "t2f"]  # sometimes named like this in newer sets
ALT_MODS = ["t1", "t1ce", "t2", "flair"]  # older naming


def find_modality_file(patient_dir: Path, mod: str) -> Path | None:
    """
    Finds a .nii or .nii.gz that contains the modality string.
    """
    candidates = list(patient_dir.rglob("*.nii")) + list(patient_dir.rglob("*.nii.gz"))
    mod_lower = mod.lower()
    for c in candidates:
        name = c.name.lower()
        if mod_lower in name:
            return c
    return None


def load_nifti(fp: Path):
    img = nib.load(str(fp))
    data = img.get_fdata().astype(np.float32)
    return img, data


def summarize(vol: np.ndarray):
    return {
        "shape": vol.shape,
        "dtype": str(vol.dtype),
        "min": float(np.min(vol)),
        "max": float(np.max(vol)),
        "mean": float(np.mean(vol)),
        "std": float(np.std(vol)),
        "p1": float(np.percentile(vol, 1)),
        "p50": float(np.percentile(vol, 50)),
        "p99": float(np.percentile(vol, 99)),
        "nonzero_%": float((np.count_nonzero(vol) / vol.size) * 100.0),
    }


def main():
    print("=== MRI Realism: Sanity Check ===")
    print("DATA_ROOT:", DATA_ROOT)
    patient_dir = DATA_ROOT / PATIENT_ID
    print("PATIENT_DIR:", patient_dir)

    if not patient_dir.exists():
        print("\n❌ Patient folder not found.")
        print("Make sure DATA_ROOT points to the directory that contains the patient folder.")
        print("Example: export MRI_DATA_ROOT=/path/to/where/BraTS-GLI-00002-000/lives")
        raise SystemExit(1)

    # Try new modality names first; if not found, try older ones.
    found_any = False
    for mod_list_name, mod_list in [("MODS", MODS), ("ALT_MODS", ALT_MODS)]:
        print(f"\n--- Searching using {mod_list_name}: {mod_list} ---")
        for mod in mod_list:
            fp = find_modality_file(patient_dir, mod)
            if fp is None:
                print(f"[{mod}]  ❌ not found")
                continue

            found_any = True
            img, vol = load_nifti(fp)
            info = summarize(vol)

            print(f"\n[{mod}] ✅ found: {fp}")
            print(" affine shape:", img.affine.shape)
            for k, v in info.items():
                print(f"  {k:10s}: {v}")

    if not found_any:
        print("\n❌ No NIfTI files found for expected modalities.")
        print("I can adapt the script if you paste `ls -R <patient_folder> | head -n 200` output.")
        raise SystemExit(1)

    print("\n✅ Sanity check done.")


if __name__ == "__main__":
    main()
