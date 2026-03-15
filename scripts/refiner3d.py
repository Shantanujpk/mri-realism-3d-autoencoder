import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Refiner3D(nn.Module):
    def __init__(self, in_ch=4, base=32):  # reduced base for memory safety
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base)
        self.pool = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base, base * 2)

        self.mid = ConvBlock(base * 2, base * 2)

        self.up = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)

        self.dec = ConvBlock(base * 2, base)

        self.out = nn.Conv3d(base, in_ch, kernel_size=1)

    def forward(self, x):

        x1 = self.enc1(x)

        x2 = self.pool(x1)
        x2 = self.enc2(x2)

        x_mid = self.mid(x2)

        x_up = self.up(x_mid)

        # Ensure shapes match
        if x_up.shape != x1.shape:
            min_d = min(x_up.shape[2], x1.shape[2])
            min_h = min(x_up.shape[3], x1.shape[3])
            min_w = min(x_up.shape[4], x1.shape[4])
            x_up = x_up[:, :, :min_d, :min_h, :min_w]
            x1 = x1[:, :, :min_d, :min_h, :min_w]

        x_cat = torch.cat([x_up, x1], dim=1)

        x_dec = self.dec(x_cat)

        residual = self.out(x_dec)

        return x + residual
