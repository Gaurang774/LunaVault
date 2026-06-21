# ============================================================
# LunaVault — Subsurface ice detection on real Chandrayaan-2 data
# ============================================================
# Data: DFSAR Level-3C polar mosaic (South Pole). Three parameters,
# defined by ISRO's official User Guide:
#   CPR  (Circular Polarization Ratio):
#        high (>1) => many radar bounces => volume scattering => ICE-like.
#        BUT rough rock also bounces a lot => high CPR too (the ambiguity).
#   SERD (Single-bounce Eigenvalue Relative Difference):
#        LOW  = rough surface ;  HIGH = smooth surface.
#        Ice deposits are smoother than blocky rock, so HIGH SERD
#        is used to reject the rough-rock false positives.
#   TRT  (T-Ratio): tracks dielectric constant.
#        LOW TRT  = low dielectric (~ice, dielectric ~3)
#        HIGH TRT = high dielectric (~rock, dielectric ~8).
#
# DETECTION (locked):
#   Candidate ice      = CPR > 1  AND  SERD > 0.75   (smooth, volume-scattering)
#   High-confidence ice= candidate AND TRT < 0.5     (also low-dielectric, pure-ice signature)
#
# Large rasters are read in row-chunks to keep memory low.
import os, glob, numpy as np, rasterio, matplotlib.pyplot as plt

BASE = "/home/gaurang/Downloads/ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx/data/derived/20250630"
def find(p): return glob.glob(f"{BASE}/*_{p}_*.tif")[0]

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# --- Locked thresholds (per ISRO User Guide + data-driven calibration) ---
CPR_T  = 1.0     # volume scattering
SERD_T = 0.75    # smooth surface (rejects rough rock); ~80th percentile of high-CPR pixels
TRT_T  = 0.5     # low dielectric (pure-ice signature) for the high-confidence subset

cpr_src = rasterio.open(find("cpr"))
srd_src = rasterio.open(find("srd"))
trt_src = rasterio.open(find("trt"))
profile = cpr_src.profile

H, W = cpr_src.height, cpr_src.width
ice_mask  = np.zeros((H, W), dtype=np.uint8)   # 1 = candidate, 2 = high-confidence

n_valid = n_cpr = n_cand = n_high = 0
CHUNK = 1000

for r0 in range(0, H, CHUNK):
    r1 = min(r0 + CHUNK, H)
    win = ((r0, r1), (0, W))
    cpr = cpr_src.read(1, window=win).astype("float32")
    srd = srd_src.read(1, window=win).astype("float32")
    trt = trt_src.read(1, window=win).astype("float32")

    valid = np.isfinite(cpr) & np.isfinite(srd) & np.isfinite(trt) & (cpr > 0)
    cpr_hit   = valid & (cpr > CPR_T)
    candidate = cpr_hit & (srd > SERD_T)                 # ice-consistent
    high_conf = candidate & (trt < TRT_T)                # strongest ice signature

    block = ice_mask[r0:r1]
    block[candidate] = 1
    block[high_conf] = 2   # overwrite candidates that are also high-confidence

    n_valid += int(valid.sum())
    n_cpr   += int(cpr_hit.sum())
    n_cand  += int(candidate.sum())
    n_high  += int(high_conf.sum())
    del cpr, srd, trt, valid, cpr_hit, candidate, high_conf

print("Valid pixels:        ", n_valid)
print("CPR>1 only:          ", n_cpr)
print(f"Candidate ice (CPR>{CPR_T} & SERD>{SERD_T}):     ", n_cand)
print(f"High-confidence (+ TRT<{TRT_T}):                 ", n_high)

prof = profile.copy(); prof.update(dtype="uint8", count=1)
with rasterio.open("data/processed/real_ice_mask.tif", "w", **prof) as dst:
    dst.write(ice_mask, 1)
print("Saved data/processed/real_ice_mask.tif  (1=candidate, 2=high-confidence)")

# Preview figure
s = 20
small = cpr_src.read(1, out_shape=(H//s, W//s)).astype("float32")
m = ice_mask[::s, ::s][:small.shape[0], :small.shape[1]]
plt.figure(figsize=(8, 8))
plt.imshow(np.clip(small, 0, 2), cmap="gray")
plt.imshow(np.ma.masked_where(m != 1, m), cmap="cool")     # candidates (cyan)
plt.imshow(np.ma.masked_where(m != 2, m), cmap="autumn")   # high-confidence (red/yellow)
plt.title(f"Ice detection: {n_cand} candidate, {n_high} high-confidence")
plt.savefig("outputs/figures/04_fusion_ice.png", dpi=120)
print("Saved outputs/figures/04_fusion_ice.png")

cpr_src.close(); srd_src.close(); trt_src.close()