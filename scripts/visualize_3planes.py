#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def window01(x, p1=1, p99=99):
    lo, hi = np.percentile(x, [p1, p99])
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x

def main():
    # Point to a single 3D nifti (one modality or 4ch)
    nifti_path = os.environ.get("NIFTI_PATH", "")
    out_png = os.environ.get("OUT_PNG", "three_planes.png")

    if not nifti_path:
        raise RuntimeError("Set NIFTI_PATH to a .nii.gz file")

    nifti_path = Path(nifti_path)
    img = nib.load(str(nifti_path)).get_fdata().astype(np.float32)

    # If it’s 4-channel nifti (H,W,D,4) or (4,H,W,D), pick one channel for display
    # We usually show ONE modality (t1c OR t2f) for clarity in the 3-plane figure.
    if img.ndim == 4:
        # Try common layouts
        if img.shape[0] == 4:      # (4,H,W,D)
            img = img[1]           # pick channel 1 (often t1c in your stack)
        elif img.shape[-1] == 4:   # (H,W,D,4)
            img = img[..., 1]
        else:
            img = img[..., 0]

    # Ensure shape is (H,W,D)
    if img.ndim != 3:
        raise RuntimeError(f"Unexpected nifti shape: {img.shape}")

    H, W, D = img.shape
    x0, y0, z0 = H // 2, W // 2, D // 2

    axial   = img[:, :, z0]      # (H,W)
    coronal = img[:, y0, :]      # (H,D)
    sagitt = img[x0, :, :]       # (W,D)

    # Window each view separately (looks much sharper/cleaner)
    axial   = window01(axial)
    coronal = window01(coronal)
    sagitt = window01(sagitt)

    plt.figure(figsize=(12, 4))

    # CRITICAL: interpolation='none' => no smoothing blur
    plt.subplot(1, 3, 1)
    plt.imshow(axial.T, cmap="gray", origin="lower", interpolation="none")
    plt.title("Axial (mid-slice)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(coronal.T, cmap="gray", origin="lower", interpolation="none")
    plt.title("Coronal (mid-slice)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sagitt.T, cmap="gray", origin="lower", interpolation="none")
    plt.title("Sagittal (mid-slice)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")  # high DPI = crisp
    print("✅ saved:", out_png)

if __name__ == "__main__":
    main()
