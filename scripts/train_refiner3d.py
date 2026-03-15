import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from scripts.dataset_gli import GLIDataset
from scripts.train_autoencoder3d import Autoencoder3D
from scripts.refiner3d import Refiner3D


# -----------------------------
# Patch Cropping Function
# -----------------------------
def random_patch(x, patch_size=(96, 128, 128)):
    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size

    d0 = torch.randint(0, D - pd, (1,)).item()
    h0 = torch.randint(0, H - ph, (1,)).item()
    w0 = torch.randint(0, W - pw, (1,)).item()

    return x[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw]


# -----------------------------
# Gradient Loss
# -----------------------------
def gradient_loss(pred, target):
    dx_pred = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    dx_target = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])

    dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dy_target = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])

    dz_pred = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
    dz_target = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])

    return (
        F.l1_loss(dx_pred, dx_target) +
        F.l1_loss(dy_pred, dy_target) +
        F.l1_loss(dz_pred, dz_target)
    )


# -----------------------------
# Main Training
# -----------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.environ["MRI_DATA_ROOT"]
    ae_ckpt_path = os.environ["AE_CKPT"]
    outdir = Path(os.environ["REFINER_OUTDIR"])
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = GLIDataset(data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # ---- Load Autoencoder ----
    ae = Autoencoder3D(base=64).to(device)
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    for p in ae.parameters():
        p.requires_grad = False

    # ---- Refiner ----
    refiner = Refiner3D(in_ch=4, base=32).to(device)

    optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda")

    epochs = 5

    for epoch in range(epochs):

        total_loss = 0

        for x, _ in loader:

            x = x.to(device)

            with torch.no_grad():
                recon = ae(x)

            # PATCH TRAINING (memory safe)
            x_patch = random_patch(x)
            recon_patch = random_patch(recon)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                refined = refiner(recon_patch)

                l1 = F.l1_loss(refined, x_patch)
                grad = gradient_loss(refined, x_patch)

                loss = l1 + 0.5 * grad

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")

        torch.save(
            {"model": refiner.state_dict()},
            outdir / f"refiner_epoch{epoch}.pt"
        )

    print("Refiner training complete.")


if __name__ == "__main__":
    main()
