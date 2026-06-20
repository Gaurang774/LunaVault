# Builds fake test data so the detection code can be run without the real
# satellite files. It makes up a 400x400 scene with mostly background, then
# plants one patch that looks like ice and one rough patch, turns those into
# the four Stokes bands, and saves them as test_S1..S4 GeoTIFFs.
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

os.makedirs("data/raw/dfsar", exist_ok=True)

H, W = 400, 400
rng = np.random.default_rng(42)

S1 = rng.uniform(0.8, 1.2, (H, W))
cpr_base = rng.uniform(0.2, 0.6, (H, W))
dop_base = rng.uniform(0.4, 0.9, (H, W))

yy, xx = np.mgrid[0:H, 0:W]
ice = (xx-260)**2 + (yy-150)**2 < 35**2
cpr_base[ice] = rng.uniform(1.1, 1.6, ice.sum())
dop_base[ice] = rng.uniform(0.04, 0.12, ice.sum())

rough = (xx-120)**2 + (yy-300)**2 < 30**2
cpr_base[rough] = rng.uniform(1.1, 1.5, rough.sum())
dop_base[rough] = rng.uniform(0.5, 0.8, rough.sum())

S4 = S1 * (1 - cpr_base) / (1 + cpr_base)
rem = np.clip((dop_base*S1)**2 - S4**2, 0, None)
S2 = np.sqrt(rem) * rng.uniform(0.3, 0.7, (H, W))
S3 = np.sqrt(np.clip(rem - S2**2, 0, None))

transform = from_origin(0, 0, 1, 1)
profile = dict(driver="GTiff", height=H, width=W, count=1,
               dtype="float32", crs="EPSG:4326", transform=transform)

for name, arr in [("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4)]:
    with rasterio.open(f"data/raw/dfsar/test_{name}.tif", "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)

print("Wrote test_S1..S4.tif to data/raw/dfsar/")