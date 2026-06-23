# ============================================================
# LunaVault — Principled Threshold Selection via PSR Enrichment
# ============================================================
# Instead of an arbitrary SERD cutoff, we choose the threshold that
# MAXIMISES how concentrated ice detections are inside cold traps
# (permanently shadowed regions), relative to the random baseline.
#
# Enrichment = (fraction of detected ice in PSR) / (PSR fraction of scene)
#   = 1.0 -> detections are random w.r.t. shadow
#   > 1.0 -> detections are physically enriched in cold traps (good)
#
# Inputs: the three DFSAR rasters + the PSR mask reprojected to grid.
# Output: enrichment vs SERD curve + chosen threshold.
# ============================================================
import glob, numpy as np, rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

BASE = "/home/gaurang/Downloads/ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx/data/derived/20250630"
def find(p): return glob.glob(f"{BASE}/*_{p}_*.tif")[0]

CPR_T = 1.0
SERD_SWEEP = np.arange(0.50, 0.96, 0.05)   # thresholds to test

# ============================================================
# 1. Load CPR + SERD (full), and reproject PSR onto their grid
# ============================================================
print("Loading CPR, SERD...")
cpr_src = rasterio.open(find("cpr"))
srd_src = rasterio.open(find("srd"))
H, W = cpr_src.height, cpr_src.width
cpr = cpr_src.read(1).astype("float32")
srd = srd_src.read(1).astype("float32")
ice_transform = cpr_src.profile["transform"]
ice_crs = cpr_src.profile["crs"]

print("Reprojecting PSR onto radar grid...")
psr = np.zeros((H, W), dtype=np.float32)
with rasterio.open("data/raw/masks/LPSR_80S_20MPP_ADJ.TIF") as ps:
    reproject(source=rasterio.band(ps, 1), destination=psr,
              src_transform=ps.transform, src_crs=ps.crs,
              dst_transform=ice_transform, dst_crs=ice_crs,
              resampling=Resampling.nearest)
psr_bin = (psr >= 0.5)

# ============================================================
# 2. Baseline: what fraction of VALID radar scene is in PSR?
# ============================================================
valid = np.isfinite(cpr) & np.isfinite(srd) & (cpr > 0)
scene_psr_frac = psr_bin[valid].mean()
print(f"\nBaseline: PSR covers {100*scene_psr_frac:.1f}% of the valid radar scene")
print(f"(random detections would hit PSR ~{100*scene_psr_frac:.1f}% of the time)\n")

# ============================================================
# 3. Sweep SERD, measure enrichment at each threshold
# ============================================================
cpr_hit = valid & (cpr > CPR_T)
print(f"{'SERD':>6}{'#ice':>10}{'%in PSR':>10}{'enrichment':>12}")
results = []
for s in SERD_SWEEP:
    ice = cpr_hit & (srd > s)
    n = int(ice.sum())
    if n == 0:
        continue
    in_psr = psr_bin[ice].mean()
    enr = in_psr / scene_psr_frac if scene_psr_frac > 0 else 0
    results.append((s, n, in_psr, enr))
    print(f"{s:>6.2f}{n:>10,}{100*in_psr:>9.1f}%{enr:>11.2f}x")

# ============================================================
# 4. Choose threshold = peak enrichment (with enough pixels)
# ============================================================
# Require at least 500 pixels so we don't pick a tiny noisy tail.
viable = [r for r in results if r[1] >= 500]
best = max(viable, key=lambda r: r[3])
print(f"\n=> Best threshold: SERD > {best[0]:.2f}")
print(f"   {best[1]:,} ice pixels, {100*best[2]:.1f}% in PSR, "
      f"{best[3]:.2f}x enrichment over baseline")

# ============================================================
# 5. Plot enrichment + pixel-count vs threshold
# ============================================================
rs = np.array(results)
fig, ax1 = plt.subplots(figsize=(9,5.5))
ax1.plot(rs[:,0], rs[:,3], "o-", color="#0e6b63", lw=2, label="enrichment")
ax1.axhline(1.0, ls="--", color="gray", label="random baseline (1.0x)")
ax1.axvline(best[0], ls=":", color="#b86b00", label=f"chosen ({best[0]:.2f})")
ax1.set_xlabel("SERD threshold")
ax1.set_ylabel("PSR enrichment (x baseline)", color="#0e6b63")
ax1.legend(loc="upper left", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(rs[:,0], rs[:,1], "s--", color="#888", alpha=.6)
ax2.set_ylabel("ice pixel count", color="#888")
ax2.set_yscale("log")
plt.title("Choosing SERD by PSR enrichment\n(higher = detections more concentrated in cold traps)")
plt.tight_layout()
plt.savefig("outputs/figures/08_threshold_enrichment.png", dpi=130)
print("\nSaved: outputs/figures/08_threshold_enrichment.png")

cpr_src.close(); srd_src.close()