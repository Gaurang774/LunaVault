# ============================================================
# LunaVault — Detection v2: CPR>1 restricted to PSRs
# ============================================================
# CORRECTION: Our earlier SERD>0.75 filter excluded the actual
# ice-bearing craters (F2, F3, S1), which are NOT smooth. Following
# Sinha et al. (2026), the primary ice indicator is CPR>1 within
# permanently shadowed regions. We report SERD/TRT as SECONDARY
# context, not as hard filters.
#
# Output: data/processed/ice_cpr_psr.tif
#         (1 = CPR>1 in PSR;  2 = also smooth+low-dielectric subset)
# ============================================================
import os, glob, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

BASE = "/home/gaurang/Downloads/ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx/data/derived/20250630"
def find(p): return glob.glob(f"{BASE}/*_{p}_*.tif")[0]

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

CPR_T = 1.0

# ---- open rasters ----
cpr_src = rasterio.open(find("cpr"))
srd_src = rasterio.open(find("srd"))
trt_src = rasterio.open(find("trt"))
profile = cpr_src.profile
H, W = cpr_src.height, cpr_src.width
transform = cpr_src.transform
crs = cpr_src.crs
px = abs(transform.a)
print(f"Grid: {W}x{H}, {px:.0f} m/px")

# ---- reproject PSR mask onto this grid ----
print("Reprojecting PSR mask...")
psr = np.zeros((H, W), dtype=np.float32)
with rasterio.open("data/raw/masks/LPSR_80S_20MPP_ADJ.TIF") as ps:
    reproject(source=rasterio.band(ps, 1), destination=psr,
              src_transform=ps.transform, src_crs=ps.crs,
              dst_transform=transform, dst_crs=crs,
              resampling=Resampling.nearest)
psr_bin = (psr >= 0.5)
print(f"PSR pixels on grid: {int(psr_bin.sum()):,}")

# ---- detect in chunks ----
ice_mask = np.zeros((H, W), dtype=np.uint8)
n_valid = n_cpr = n_cpr_psr = n_secondary = 0
CHUNK = 1000
for r0 in range(0, H, CHUNK):
    r1 = min(r0 + CHUNK, H)
    win = ((r0, r1), (0, W))
    cpr = cpr_src.read(1, window=win).astype("float32")
    srd = srd_src.read(1, window=win).astype("float32")
    trt = trt_src.read(1, window=win).astype("float32")
    pslice = psr_bin[r0:r1]

    valid = np.isfinite(cpr) & np.isfinite(srd) & np.isfinite(trt) & (cpr > 0)
    cpr_hit = valid & (cpr > CPR_T)
    in_psr  = cpr_hit & pslice                       # PRIMARY: CPR>1 in cold trap
    secondary = in_psr & (srd > 0.6) & (trt < 0.5)   # optional smooth+low-diel subset

    block = ice_mask[r0:r1]
    block[in_psr] = 1
    block[secondary] = 2

    n_valid     += int(valid.sum())
    n_cpr       += int(cpr_hit.sum())
    n_cpr_psr   += int(in_psr.sum())
    n_secondary += int(secondary.sum())
    del cpr, srd, trt, valid, cpr_hit, in_psr, secondary

print("\n===== DETECTION v2 (CPR>1 in PSR) =====")
print(f"Valid pixels:              {n_valid:,}")
print(f"CPR>1 (all):               {n_cpr:,}")
print(f"CPR>1 inside PSR (ICE):    {n_cpr_psr:,}")
print(f"  + smooth+low-diel subset:{n_secondary:,}")

# ---- save ----
prof = profile.copy(); prof.update(dtype="uint8", count=1)
with rasterio.open("data/processed/ice_cpr_psr.tif", "w", **prof) as dst:
    dst.write(ice_mask, 1)
print("\nSaved: data/processed/ice_cpr_psr.tif (1=CPR>1 in PSR, 2=secondary subset)")

# ---- preview figure ----
s = 20
small = cpr_src.read(1, out_shape=(H//s, W//s)).astype("float32")
m = ice_mask[::s, ::s][:small.shape[0], :small.shape[1]]
plt.figure(figsize=(8, 8))
plt.imshow(np.clip(small, 0, 2), cmap="gray")
plt.imshow(np.ma.masked_where(m == 0, m), cmap="cool")
plt.title(f"Detection v2: CPR>1 in PSR ({n_cpr_psr:,} px)")
plt.savefig("outputs/figures/09_cpr_psr_detection.png", dpi=120)
print("Saved: outputs/figures/09_cpr_psr_detection.png")

cpr_src.close(); srd_src.close(); trt_src.close()