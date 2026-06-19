import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, binary_closing, disk

CPR_T = 1.0
DOP_T = 0.13
MIN_SIZE = 5
FOLDER = "data/raw/dfsar"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.profile

def save_raster(arr, profile, path):
    profile = profile.copy()
    profile.update(dtype="float32", count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    print("Saved", path)

# Load Stokes
S = {}
for k in ["S1", "S2", "S3", "S4"]:
    S[k], profile = load_raster(f"{FOLDER}/test_{k}.tif")
S1, S2, S3, S4 = S["S1"], S["S2"], S["S3"], S["S4"]
eps = 1e-6

cpr = (S1 - S4) / (S1 + S4 + eps)
dop = np.sqrt(S2**2 + S3**2 + S4**2) / (S1 + eps)

ice = (cpr > CPR_T) & (dop < DOP_T)
ice = remove_small_objects(ice, min_size=MIN_SIZE)
ice = binary_closing(ice, disk(2))

save_raster(cpr, profile, "data/processed/cpr.tif")
save_raster(dop, profile, "data/processed/dop.tif")
save_raster(ice.astype("float32"), profile, "data/processed/ice_mask.tif")

print(f"Ice pixels detected: {int(ice.sum())}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
im0 = ax[0].imshow(cpr, cmap="jet", vmin=0, vmax=2); ax[0].set_title("CPR"); plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(dop, cmap="viridis", vmin=0, vmax=1); ax[1].set_title("DOP"); plt.colorbar(im1, ax=ax[1])
ax[2].imshow(S1, cmap="gray")
ax[2].imshow(np.ma.masked_where(~ice, ice), cmap="autumn", alpha=0.9)
ax[2].set_title(f"Ice mask ({int(ice.sum())} px)")
plt.tight_layout()
plt.savefig("outputs/figures/02_ice_detection.png", dpi=120)
print("Saved outputs/figures/02_ice_detection.png")