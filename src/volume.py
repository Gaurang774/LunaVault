# Rough estimate of how much ice the detected area could hold. It reads the
# ice mask, works out the real-world area of the flagged pixels, then guesses
# the tonnage many times over with random depth and ice-fraction values. The
# spread of those guesses is reported as low, middle and high (P10/P50/P90)
# numbers and written to a small JSON report.
import numpy as np
import rasterio
import json

with rasterio.open("data/processed/real_ice_mask.tif") as src:
    ice = src.read(1)
    px_area = abs(src.transform[0] * src.transform[4])  # m² per pixel

ice_px = int((ice > 0).sum())
area = ice_px * px_area  # m²

rng = np.random.default_rng(0)
depth = rng.uniform(0, 5, 100000)        # 0–5 m
frac = rng.uniform(0.02, 0.10, 100000)   # 2–10% ice
rho = 917                                # kg/m³ (ice density)

vol_kg = area * depth * frac * rho
tonnes = vol_kg / 1000

p10, p50, p90 = np.percentile(tonnes, [10, 50, 90])

report = {
    "ice_pixels": ice_px,
    "area_m2": round(area, 1),
    "P10_tonnes": round(float(p10), 1),
    "P50_tonnes": round(float(p50), 1),
    "P90_tonnes": round(float(p90), 1),
}
print(report)

json.dump(report, open("data/processed/volume_report.json", "w"), indent=2)
print("Saved volume_report.json")