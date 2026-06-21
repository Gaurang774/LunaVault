import glob, numpy as np, rasterio

BASE = "/home/gaurang/Downloads/ch2_sar_ndxl_20250630mpcpspeast_d_fp_xxx/data/derived/20250630"
def find(p): return glob.glob(f"{BASE}/*_{p}_*.tif")[0]

cpr_src = rasterio.open(find("cpr"))
srd_src = rasterio.open(find("srd"))
trt_src = rasterio.open(find("trt"))
H, W = cpr_src.height, cpr_src.width

# Sample only high-CPR pixels and look at their SERD/TRT
srd_vals, trt_vals = [], []
CHUNK = 2000
for r0 in range(0, H, CHUNK):
    r1 = min(r0+CHUNK, H)
    win = ((r0, r1), (0, W))
    cpr = cpr_src.read(1, window=win).astype("float32")
    srd = srd_src.read(1, window=win).astype("float32")
    trt = trt_src.read(1, window=win).astype("float32")
    m = np.isfinite(cpr)&np.isfinite(srd)&np.isfinite(trt)&(cpr>1.0)
    srd_vals.append(srd[m]); trt_vals.append(trt[m])
    del cpr,srd,trt,m

srd_vals = np.concatenate(srd_vals)
trt_vals = np.concatenate(trt_vals)

print("Among CPR>1 pixels (", len(srd_vals), "total):")
print("  SERD: min %.2f  median %.2f  max %.2f" % (srd_vals.min(), np.median(srd_vals), srd_vals.max()))
print("    SERD percentiles 25/50/75/90:", np.round(np.percentile(srd_vals,[25,50,75,90]),2))
print("  TRT:  min %.2f  median %.2f  max %.2f" % (trt_vals.min(), np.median(trt_vals), trt_vals.max()))
print("    TRT percentiles 25/50/75/90:", np.round(np.percentile(trt_vals,[25,50,75,90]),2))
print("  How many CPR>1 also have SERD>0.75:", int((srd_vals>0.75).sum()))
print("  How many CPR>1 also have TRT<0.5:", int((trt_vals<0.5).sum()))
print("  How many have BOTH SERD>0.75 AND TRT<0.5:", int(((srd_vals>0.75)&(trt_vals<0.5)).sum()))
cpr_src.close(); srd_src.close(); trt_src.close()