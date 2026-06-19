import rasterio
import numpy as np

def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
    return arr, profile

def load_stokes(folder, prefix="test_"):
    s = {}
    for k in ["S1", "S2", "S3", "S4"]:
        arr, prof = load_raster(f"{folder}/{prefix}{k}.tif")
        s[k] = arr
    s["profile"] = prof
    return s

def save_raster(arr, profile, path):
    profile = profile.copy()
    profile.update(dtype="float32", count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    print("Saved", path)