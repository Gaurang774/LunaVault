import numpy as np, rasterio, json

with rasterio.open("data/processed/ice_mask.tif") as src:
    ice = src.read(1)
    px_area = abs(src.transform[0] * src.transform[4])  # m² per pixel

ice_px = int((ice > 0).sum())
area = ice_px * px_area  # m²

rng = np.random.default_rng(0)
depth = rng.uniform(0, 5, 100000)            # 0–5 m
frac  = rng.uniform(0.02, 0.10, 100000)      # 2–10% ice
rho   = 917                                   # kg/m³
vol_kg = area * depth * frac * rho
tonnes = vol_kg / 1000

p10, p50, p90 = np.percentile(tonnes, [10, 50, 90])
report = {"ice_pixels": ice_px, "area_m2": area,
          "P10_t": p10, "P50_t": p50, "P90_t": p90}
print(report)
json.dump(report, open("data/processed/volume_report.json", "w"), indent=2)
print("Saved volume_report.json")