import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch


# ----------------------------
# Helpers
# ----------------------------
MODS = ["t1n", "t1c", "t2w", "t2f"]

def find_mod_file(patient_dir: Path, pid: str, mod: str) -> Path:
    # BraTS-GLI-xxxxx-xxx-t1n.nii.gz style
    p = patient_dir / f"{pid}-{mod}.nii.gz"
    if p.exists():
        return p
    # fallback: try any file containing "-{mod}.nii.gz"
    candidates = list(patient_dir.glob(f"*{mod}.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find modality {mod} in {patient_dir}")

def load_4ch_volume(data_root: Path, patient_id: str):
    patient_dir = data_root / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient dir not found: {patient_dir}")

    vols = []
    affine = None

    for mod in MODS:
        f = find_mod_file(patient_dir, patient_id, mod)
        img = nib.load(str(f))
        arr = img.get_fdata(dtype=np.float32)
        vols.append(arr)
        if affine is None:
            affine = img.affine

    # shape: (4, H, W, D)
    x = np.stack(vols, axis=0)
    return x, affine

def normalize_like_training(x: np.ndarray) -> np.ndarray:
    """
    Keep it simple & stable:
    - clip each channel to its 99th percentile
    - scale to [0,1]
    """
    x = x.copy()
    for c in range(x.shape[0]):
        p99 = np.percentile(x[c], 99.0)
        if p99 <= 0:
            p99 = 1.0
        x[c] = np.clip(x[c], 0, p99) / p99
    return x

def denorm_to_float(x_norm: np.ndarray) -> np.ndarray:
    """
    We don't know original scale per patient after normalization.
    So we save recon in normalized float32 in [0,1].
    That's perfectly valid for visualization + comparison.
    """
    return x_norm.astype(np.float32)

def load_model_from_training_script(ckpt_path: Path, device: str):
    # Your model class is defined inside train_autoencoder3d.py
    from scripts.train_autoencoder3d import Autoencoder3D

    model = Autoencoder3D(in_ch=4, base=32).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # handle common checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unknown checkpoint format")

    # strip possible "module." prefix
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


# ----------------------------
# Main
# ----------------------------
def main():
    data_root = Path(os.environ["MRI_DATA_ROOT"]).expanduser()
    patient_id = os.environ["PATIENT_ID"]
    outdir = Path(os.environ["OUTDIR"]).expanduser()
    ckpt_path = Path(os.environ["CKPT"]).expanduser()

    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("DATA_ROOT:", data_root)
    print("PATIENT:", patient_id)
    print("CKPT:", ckpt_path)
    print("OUTDIR:", outdir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load patient 3D volumes (4ch) + affine
    x_np, affine = load_4ch_volume(data_root, patient_id)
    x_norm = normalize_like_training(x_np)

    # Torch tensor: (B, C, H, W, D)
    x = torch.from_numpy(x_norm).unsqueeze(0).to(device)

    # Load model + reconstruct
    model = load_model_from_training_script(ckpt_path, device)

    with torch.no_grad():
        y = model(x)  # (B,C,H,W,D)

    y_np = y.squeeze(0).detach().cpu().numpy()
    y_np = np.clip(y_np, 0.0, 1.0)

    # Save as 4D NIfTI: (H,W,D,C)
    # NIfTI convention: spatial dims first
    x_save = np.transpose(denorm_to_float(x_norm), (1, 2, 3, 0))
    y_save = np.transpose(denorm_to_float(y_np), (1, 2, 3, 0))

    x_nii = nib.Nifti1Image(x_save, affine)
    y_nii = nib.Nifti1Image(y_save, affine)

    out_x = outdir / f"input_{patient_id}_4ch.nii.gz"
    out_y = outdir / f"recon_{patient_id}_4ch.nii.gz"

    nib.save(x_nii, str(out_x))
    nib.save(y_nii, str(out_y))

    # Also save each modality separately (easier to view)
    for i, mod in enumerate(MODS):
        nib.save(nib.Nifti1Image(x_save[..., i], affine), str(outdir / f"input_{patient_id}_{mod}.nii.gz"))
        nib.save(nib.Nifti1Image(y_save[..., i], affine), str(outdir / f"recon_{patient_id}_{mod}.nii.gz"))

    print("✅ Saved 3D NIfTI reconstructions:")
    print(" -", out_x)
    print(" -", out_y)
    print("✅ Saved per-modality 3D NIfTI files too.")


if __name__ == "__main__":
    main()
