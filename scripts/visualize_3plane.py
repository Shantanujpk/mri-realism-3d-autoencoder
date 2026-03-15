import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def normalize_slice(x):
    x = np.nan_to_num(x)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return x
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_3plane.py <path_to_nii.gz> [out.png]")
        sys.exit(1)

    nii_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) >= 3 else "3plane_view.png"

    img = nib.load(nii_path)
    data = img.get_fdata()

    # data is typically (H, W, D)
    H, W, D = data.shape

    axial = data[:, :, D // 2]
    coronal = data[:, W // 2, :]
    sagittal = data[H // 2, :, :]

    axial = normalize_slice(axial)
    coronal = normalize_slice(coronal)
    sagittal = normalize_slice(sagittal)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np.rot90(axial), cmap="gray")
    axes[0].set_title("Axial (mid-slice)")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(coronal), cmap="gray")
    axes[1].set_title("Coronal (mid-slice)")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(sagittal), cmap="gray")
    axes[2].set_title("Sagittal (mid-slice)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"✅ Saved: {out_png}")

if __name__ == "__main__":
    main()
