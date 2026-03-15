import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from scripts.dataset_gli import GLIDataset


# ===============================
# BIG 3D AUTOENCODER
# ===============================

class Autoencoder3D(nn.Module):
    def __init__(self, base=96):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv3d(4, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),

            nn.Conv3d(base, base, 4, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),

            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),

            nn.Conv3d(base * 2, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),

            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.SiLU(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose3d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),

            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),

            nn.ConvTranspose3d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),

            nn.Conv3d(base, 4, 3, padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out


# ===============================
# TRAINING
# ===============================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = os.environ["MRI_DATA_ROOT"]
    outdir = os.environ["OUTDIR"]
    resume_path = os.environ.get("RESUME", None)

    os.makedirs(outdir, exist_ok=True)

    dataset = GLIDataset(data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    model = Autoencoder3D(base=96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = GradScaler()

    start_epoch = 0
    num_epochs = 20

    # Resume support
    if resume_path and os.path.exists(resume_path):
        print("Resuming from:", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, num_epochs):

        model.train()
        total_loss = 0

        for batch in loader:

            # Safe dataset handling
            if isinstance(batch, dict):
                x = batch.get("image", list(batch.values())[0])
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)

            optimizer.zero_grad()

            with autocast():
                recon = model(x)

                # 🔥 FIX DEPTH MISMATCH HERE
                min_d = min(recon.shape[-1], x.shape[-1])
                recon = recon[..., :min_d]
                x_cropped = x[..., :min_d]

                loss = F.l1_loss(recon, x_cropped)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"\n=== Epoch {epoch} | Avg Loss: {avg_loss:.6f} ===", flush=True)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, os.path.join(outdir, f"checkpoint_epoch{epoch}.pt"))

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
