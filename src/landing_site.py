# ============================================================
# LunaVault — Landing Site Selection (Official Step 4)
# MEMORY-SAFE version: works on a downsampled grid to avoid crashes.
# ============================================================
# Finds safe, flat terrain near PSR-restricted ice deposits.
# All inputs reprojected onto the (downsampled) ice grid so they align.
# Inputs:
#   data/processed/ice_in_psr.tif        (cold-trap ice pixels)
#   data/raw/dtm/LDEM_80S_80MPP_ADJ.TIF  (elevation, Moon-2015)
#   data/raw/dtm/LDSM_80S_80MPP_ADJ.TIF  (slope deg, Moon-2015)
# Outputs:
#   outputs/figures/06_landing_sites.png
#   prints top 5 candidate landing sites
# ============================================================
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RS
from scipy import ndimage
import matplotlib.pyplot as plt
from pyproj import Transformer

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# --- Safety criteria ---
MAX_SLOPE_DEG   = 15.0    # lander tip-over limit
MAX_ICE_DIST_KM = 10.0    # rover lands within 10 km of ice
DOWNSAMPLE      = 4       # process at 4x coarser res to save memory

# ============================================================
# 1. Load ice mask DOWNSAMPLED (defines target grid)
# ============================================================
with rasterio.open("data/processed/ice_in_psr.tif") as src:
    H0, W0 = src.height, src.width
    H, W = H0 // DOWNSAMPLE, W0 // DOWNSAMPLE
    # read at reduced resolution
    # build the downsampled transform
    ice = src.read(1, out_shape=(H, W), resampling=RS.nearest)  # keep ice presence
    ice_transform = src.transform * src.transform.scale(W0 / W, H0 / H)
    ice_crs = src.crs
    px_m = abs(ice_transform.a)
    print(f"Ice grid (downsampled): {ice.shape}, {px_m:.0f} m/px")

ice_present = (ice >= 1).astype(np.uint8)
print(f"Ice pixels (downsampled): {int(ice_present.sum())}")

# ============================================================
# 2. Reproject DEM + slope onto the downsampled ice grid
# ============================================================
def reproject_to_grid(path):
    out = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ice_transform,
            dst_crs=ice_crs,
            resampling=Resampling.bilinear,
        )
    return out

print("Reprojecting DEM...")
dem = reproject_to_grid("data/raw/dtm/LDEM_80S_80MPP_ADJ.TIF")
print("Reprojecting slope...")
slope = reproject_to_grid("data/raw/dtm/LDSM_80S_80MPP_ADJ.TIF")

valid = np.isfinite(slope) & np.isfinite(dem)
print(f"Valid terrain pixels: {int(valid.sum())}")

# ============================================================
# 3. Safe terrain = valid + low slope
# ============================================================
safe = valid & (slope < MAX_SLOPE_DEG)
print(f"Safe terrain (slope < {MAX_SLOPE_DEG} deg): {int(safe.sum())}")

# ============================================================
# 4. Distance from each pixel to nearest ice
# ============================================================
if ice_present.sum() > 0:
    dist_px = ndimage.distance_transform_edt(ice_present == 0)
    dist_km = dist_px * px_m / 1000.0
else:
    dist_km = np.full((H, W), 1e9, dtype=np.float32)
    print("WARNING: no ice pixels found at this resolution.")

# ============================================================
# 5. Score landing sites: safe AND close to ice
# ============================================================
near_ice = safe & (dist_km < MAX_ICE_DIST_KM)
print(f"Safe AND within {MAX_ICE_DIST_KM} km of ice: {int(near_ice.sum())}")

slope_score = np.clip(1.0 - slope / MAX_SLOPE_DEG, 0, 1)
dist_score  = np.clip(1.0 - dist_km / MAX_ICE_DIST_KM, 0, 1)
score = np.where(near_ice, 0.5 * slope_score + 0.5 * dist_score, 0.0)
print(f"Best score: {np.nanmax(score):.3f}")

# ============================================================
# 6. Top 5 candidate sites (spread apart)
# ============================================================
tr = Transformer.from_crs(ice_crs.to_string(),
                          "+proj=longlat +R=1737400 +no_defs",
                          always_xy=True)
def pixel_to_latlon(r, c):
    x = ice_transform.c + c * ice_transform.a + r * ice_transform.b
    y = ice_transform.f + c * ice_transform.d + r * ice_transform.e
    lon, lat = tr.transform(x, y)
    return lat, lon

print("\n===== TOP 5 CANDIDATE LANDING SITES =====")
work = score.copy()
sites = []
for i in range(5):
    r, c = np.unravel_index(np.argmax(work), work.shape)
    if work[r, c] <= 0:
        break
    lat, lon = pixel_to_latlon(r, c)
    sites.append((i+1, r, c, lat, lon, float(score[r,c]),
                  float(slope[r,c]), float(dist_km[r,c])))
    print(f"  Site {i+1}: lat {lat:.2f}, lon {lon:.2f} | "
          f"score {score[r,c]:.3f} | slope {slope[r,c]:.1f} deg | "
          f"{dist_km[r,c]:.1f} km to ice")
    rad = 15
    work[max(0,r-rad):r+rad, max(0,c-rad):c+rad] = 0

if not sites:
    print("  No valid sites found (try relaxing thresholds).")

# ============================================================
# 7. Figure
# ============================================================
plt.figure(figsize=(10, 10))
plt.imshow(np.clip(slope, 0, 30), cmap="terrain_r")
plt.colorbar(label="slope (deg)", shrink=0.7)
plt.imshow(np.ma.masked_where(ice_present == 0, ice_present), cmap="cool", alpha=0.9)
for site in sites:
    _, r, c, lat, lon, sc, sl, di = site
    plt.scatter(c, r, c="red", s=150, marker="*", edgecolors="white", zorder=5)
    plt.annotate(f"#{site[0]}", (c, r), color="white", fontsize=12, weight="bold")
plt.title(f"Landing sites (stars) near PSR ice (cyan)\n"
          f"slope < {MAX_SLOPE_DEG} deg, within {MAX_ICE_DIST_KM} km of ice")
plt.savefig("outputs/figures/06_landing_sites.png", dpi=130, bbox_inches="tight")
print("\nSaved: outputs/figures/06_landing_sites.png")