import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.dataset_gli import GLIDataset
from scripts.train_autoencoder3d import Autoencoder3D
from scripts.refiner3d import Refiner3D


def center_slices(volume):
    # volume shape: (C, D, H, W)
    _, D, H, W = volume.shape

    d_mid = D // 2
    h_mid = H // 2
    w_mid = W // 2

    axial = volume[0, d_mid, :, :]
    coronal = volume[0, :, h_mid, :]
    sagittal = volume[0, :, :, w_mid]

    return axial, coronal, sagittal


def normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.environ["MRI_DATA_ROOT"]
    ae_ckpt = os.environ["AE_CKPT"]
    refiner_ckpt = os.environ["REFINER_CKPT"]
    patient_id = os.environ["PATIENT_ID"]
    outdir = Path(os.environ["OUTDIR"])
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    dataset = GLIDataset(data_root)

    # Find patient index safely
    patient_index = None
    for i in range(len(dataset)):
        _, pid = dataset[i]
        if pid == patient_id:
            patient_index = i
            break

    if patient_index is None:
        raise ValueError(f"Patient {patient_id} not found in dataset.")

    x, _ = dataset[patient_index]
    x = x.unsqueeze(0).to(device)

    # ---- Load AE ----
    ae = Autoencoder3D(base=64).to(device)
    ckpt = torch.load(ae_ckpt, map_location=device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    # ---- Load Refiner ----
    refiner = Refiner3D(in_ch=4, base=32).to(device)
    r_ckpt = torch.load(refiner_ckpt, map_location=device)
    refiner.load_state_dict(r_ckpt["model"])
    refiner.eval()

    with torch.no_grad():
        recon = ae(x)
        refined = refiner(recon)

    x = x[0].cpu()
    recon = recon[0].cpu()
    refined = refined[0].cpu()

    # Use T1c channel (index 1)
    orig_slices = center_slices(x[1:2])
    ae_slices = center_slices(recon[1:2])
    ref_slices = center_slices(refined[1:2])

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    titles = ["Axial", "Coronal", "Sagittal"]
    rows = ["Original", "AE Recon", "Refined"]

    for col in range(3):
        axes[0, col].imshow(normalize(orig_slices[col]), cmap="gray")
        axes[1, col].imshow(normalize(ae_slices[col]), cmap="gray")
        axes[2, col].imshow(normalize(ref_slices[col]), cmap="gray")

        axes[0, col].set_title(titles[col])

    for row in range(3):
        axes[row, 0].set_ylabel(rows[row], fontsize=12)

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()

    save_path = outdir / f"compare_{patient_id}.png"
    plt.savefig(save_path, dpi=300)
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
