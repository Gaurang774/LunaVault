# Real-data ice detection using three radar products together: CPR, SRD and
# TRT. A pixel is called ice only when all three agree (high CPR, low SRD,
# high TRT). The real rasters are huge, so they are read a few thousand rows
# at a time to keep memory in check, counts are tallied as it goes, and the
# final mask is written out along with a small downsampled preview image.
import os, glob, numpy as np, rasterio, matplotlib.pyplot as plt

BASE = "/home/gaurang/Downloads/ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx/data/derived/20250630"
def find(p): return glob.glob(f"{BASE}/*_{p}_*.tif")[0]

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

CPR_T, SRD_T, TRT_T = 1.0, 0.3, 1.0

# Open all three (lazy - not loaded yet)
cpr_src = rasterio.open(find("cpr"))
srd_src = rasterio.open(find("srd"))
trt_src = rasterio.open(find("trt"))
profile = cpr_src.profile

H, W = cpr_src.height, cpr_src.width
ice_mask = np.zeros((H, W), dtype=np.uint8)

n_valid = n_cpr = n_ice = 0
CHUNK = 1000  # rows at a time

for r0 in range(0, H, CHUNK):
    r1 = min(r0 + CHUNK, H)
    win = ((r0, r1), (0, W))
    cpr = cpr_src.read(1, window=win).astype("float32")
    srd = srd_src.read(1, window=win).astype("float32")
    trt = trt_src.read(1, window=win).astype("float32")

    valid = np.isfinite(cpr) & np.isfinite(srd) & np.isfinite(trt) & (cpr > 0)
    cpr_hit = valid & (cpr > CPR_T)
    ice = cpr_hit & (srd < SRD_T) & (trt > TRT_T)

    ice_mask[r0:r1][ice] = 1
    n_valid += int(valid.sum())
    n_cpr   += int(cpr_hit.sum())
    n_ice   += int(ice.sum())
    del cpr, srd, trt, valid, cpr_hit, ice  # free memory each chunk

print("Valid pixels:", n_valid)
print("CPR>1 only:", n_cpr)
print("Fusion ICE (CPR>1 & SRD<0.5 & TRT>0.5):", n_ice)

prof = profile.copy(); prof.update(dtype="uint8", count=1)
with rasterio.open("data/processed/real_ice_mask.tif", "w", **prof) as dst:
    dst.write(ice_mask, 1)
print("Saved data/processed/real_ice_mask.tif")

# Small downsampled figure (read overview, not full res)
s = 20
small = cpr_src.read(1, out_shape=(H//s, W//s)).astype("float32")
ice_small = ice_mask[::s, ::s][:small.shape[0], :small.shape[1]]
plt.figure(figsize=(8, 8))
plt.imshow(np.clip(small, 0, 2), cmap="gray")
plt.imshow(np.ma.masked_where(ice_small == 0, ice_small), cmap="cool")
plt.title(f"Fusion ice detection ({n_ice} px)")
plt.savefig("outputs/figures/04_fusion_ice.png", dpi=120)
print("Saved outputs/figures/04_fusion_ice.png")

cpr_src.close(); srd_src.close(); trt_src.close()