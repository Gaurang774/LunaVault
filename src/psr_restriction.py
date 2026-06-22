# ============================================================
# LunaVault — PSR Restriction (Official Step 1)
# Restricts detected ice to permanently shadowed regions (cold traps)
# ============================================================
# The ice mask and PSR mask have DIFFERENT CRS and bounds, so we
# reproject the PSR onto the ice mask's exact grid before combining.
# Input:  data/processed/real_ice_mask.tif  (ice: 1=candidate, 2=high-conf)
# Input:  data/raw/masks/LPSR_80S_20MPP_ADJ.TIF  (PSR: 1=shadowed, 0=sunlit)
# Output: data/processed/ice_in_psr.tif
# ============================================================
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

os.makedirs("data/processed", exist_ok=True)

# --- Load ice mask (this defines our target grid) ---
with rasterio.open("data/processed/real_ice_mask.tif") as src:
    ice = src.read(1)
    ice_transform = src.transform
    ice_crs = src.crs
    ice_profile = src.profile
    H, W = ice.shape
    print(f"Ice mask: {ice.shape}, CRS {ice_crs.to_string()[:40]}")

# --- Reproject PSR onto the ice grid ---
# This converts PSR from its CRS/bounds to exactly match the ice mask.
psr_on_ice = np.zeros((H, W), dtype=np.float32)
with rasterio.open("data/raw/masks/LPSR_80S_20MPP_ADJ.TIF") as src:
    print(f"PSR source: {src.shape}, CRS {src.crs.to_string()[:40]}")
    reproject(
        source=rasterio.band(src, 1),
        destination=psr_on_ice,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ice_transform,
        dst_crs=ice_crs,
        resampling=Resampling.nearest,   # PSR is binary, use nearest
    )

# PSR is now aligned to the ice grid. Binarize (1 = shadowed).
psr_binary = (psr_on_ice >= 0.5).astype(np.uint8)
print(f"PSR reprojected onto ice grid. Shadowed pixels in scene: {int(psr_binary.sum())}")

# --- Apply restriction: keep ice only where PSR = 1 ---
ice_in_psr = np.where(psr_binary == 1, ice, 0).astype(np.uint8)

# --- Count results ---
cand_total   = int((ice == 1).sum())
cand_in_psr  = int((ice_in_psr == 1).sum())
high_total   = int((ice == 2).sum())
high_in_psr  = int((ice_in_psr == 2).sum())

print("\n===== PSR RESTRICTION RESULTS =====")
print(f"Candidate ice (all):       {cand_total}")
print(f"Candidate ice in PSR:      {cand_in_psr}"
      f"  ({100*cand_in_psr/cand_total:.1f}%)" if cand_total else "")
print(f"High-confidence (all):     {high_total}")
print(f"High-confidence in PSR:    {high_in_psr}")
print(f"\nInterpretation: {cand_in_psr} of {cand_total} ice-consistent pixels"
      f" fall inside permanently shadowed regions (physically valid cold traps).")

# --- Save outputs ---
out_profile = ice_profile.copy()
out_profile.update(dtype="uint8", count=1)

with rasterio.open("data/processed/ice_in_psr.tif", "w", **out_profile) as dst:
    dst.write(ice_in_psr, 1)
print("\nSaved: data/processed/ice_in_psr.tif")

with rasterio.open("data/processed/psr_on_ice_grid.tif", "w", **out_profile) as dst:
    dst.write(psr_binary, 1)
print("Saved: data/processed/psr_on_ice_grid.tif (aligned PSR, for reference)")