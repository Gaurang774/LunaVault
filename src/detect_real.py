# A first look at the real Chandrayaan-2 CPR file. It opens the CPR raster,
# prints some basic stats, cleans out the bad values, and marks everywhere CPR
# is above 1 as a possible ice spot. The full image is far too big to plot, so
# it is shrunk down before saving a side-by-side picture of the CPR and the
# candidate pixels.
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

CPR_PATH = "/home/gaurang/Downloads/ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx/data/derived/20250630/ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif"
CPR_T = 1.0

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

with rasterio.open(CPR_PATH) as src:
    cpr = src.read(1).astype("float32")
    profile = src.profile

# Clean invalid values (nodata / negatives often present in SAR)
cpr = np.where(np.isfinite(cpr), cpr, np.nan)
valid = cpr[np.isfinite(cpr)]
print("CPR stats -> min:", np.nanmin(cpr), "max:", np.nanmax(cpr),
      "mean:", round(float(np.nanmean(cpr)), 3),
      "median:", round(float(np.nanmedian(valid)), 3))
print("Pixels with CPR > 1:", int(np.nansum(cpr > CPR_T)))
print("Total valid pixels:", int(np.isfinite(cpr).sum()))

# High-CPR candidate mask
ice_candidate = (cpr > CPR_T)

# Downsample for a quick viewable figure (12k x 12k is too big to plot raw)
step = 10
cpr_small = cpr[::step, ::step]
ice_small = ice_candidate[::step, ::step]

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
im0 = ax[0].imshow(np.clip(cpr_small, 0, 2), cmap="jet")
ax[0].set_title("Real CPR (Chandrayaan-2, S. Pole)"); plt.colorbar(im0, ax=ax[0])
ax[1].imshow(np.clip(cpr_small, 0, 2), cmap="gray")
ax[1].imshow(np.ma.masked_where(~ice_small, ice_small), cmap="autumn")
ax[1].set_title("CPR > 1 candidates")
plt.tight_layout()
plt.savefig("outputs/figures/03_real_cpr.png", dpi=120)
print("Saved outputs/figures/03_real_cpr.png")