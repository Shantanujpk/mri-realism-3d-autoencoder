import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.train_autoencoder3d import Autoencoder3D
from scripts.dataset_gli import GLIDataset


def center_crop_depth(tensor, target_depth):
    current_depth = tensor.shape[-1]
    if current_depth == target_depth:
        return tensor
    start = (current_depth - target_depth) // 2
    end = start + target_depth
    return tensor[..., start:end]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_root = os.environ["MRI_DATA_ROOT"]
    outdir = os.environ["OUTDIR"]
    patient_id = os.environ["PATIENT_ID"]
    ckpt_path = os.environ["CKPT"]

    print("CKPT:", ckpt_path)
    print("PATIENT:", patient_id)

    dataset = GLIDataset(data_root)

    # Find patient
    patient_index = None
    for idx in range(len(dataset)):
        x, pid = dataset[idx]
        if pid == patient_id:
            patient_index = idx
            break

    if patient_index is None:
        raise ValueError(f"Patient {patient_id} not found")

    # x is already a Tensor
    x, _ = dataset[patient_index]
    x = x.unsqueeze(0).to(device)  # add batch dimension

    model = Autoencoder3D().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        recon = model(x)

    # Handle possible depth mismatch
    if recon.shape[-1] != x.shape[-1]:
        min_depth = min(recon.shape[-1], x.shape[-1])
        x = center_crop_depth(x, min_depth)
        recon = center_crop_depth(recon, min_depth)

    x = x.cpu()[0]       # remove batch
    recon = recon.cpu()[0]

    original = x[1].numpy()        # T1c
    reconstructed = recon[1].numpy()

    H, W, D = original.shape

    axial = D // 2
    coronal = W // 2
    sagittal = H // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original[:, :, axial], cmap="gray")
    axes[0, 0].set_title("Original - Axial")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(original[:, coronal, :], cmap="gray")
    axes[0, 1].set_title("Original - Coronal")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(original[sagittal, :, :], cmap="gray")
    axes[0, 2].set_title("Original - Sagittal")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(reconstructed[:, :, axial], cmap="gray")
    axes[1, 0].set_title("Reconstruction - Axial")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(reconstructed[:, coronal, :], cmap="gray")
    axes[1, 1].set_title("Reconstruction - Coronal")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(reconstructed[sagittal, :, :], cmap="gray")
    axes[1, 2].set_title("Reconstruction - Sagittal")
    axes[1, 2].axis("off")

    plt.tight_layout()

    save_path = os.path.join(outdir, f"ae3d_recon_{patient_id}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("Saved to:", save_path)


if __name__ == "__main__":
    main()
