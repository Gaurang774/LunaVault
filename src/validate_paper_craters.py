# ============================================================
# LunaVault — Validation against EXACT paper craters
# Sinha et al. 2026, Table 1. The four ice-bearing doubly-shadowed
# craters: F2, F3 (Faustini), H3 (Haworth), S1 (Shoemaker).
# ============================================================
import rasterio, numpy as np
from rasterio.transform import rowcol
from pyproj import Transformer
import matplotlib.pyplot as plt

MASK = "data/processed/ice_cpr_psr.tif"

# Exact coordinates from Sinha et al. 2026, Table 1
craters = {
    "F2 (Faustini, STRONG)":  (-87.39,  82.31,  1100),
    "F3 (Faustini, likely)":  (-87.31,  86.333, 700),
    "H3 (Haworth, partial)":  (-87.915, 349.058, 800),
    "S1 (Shoemaker, partial)":(-87.841, 40.078, 2980),
}

with rasterio.open(MASK) as src:
    mask = src.read(1)
    transform = src.transform
    crs = src.crs
    px = abs(transform.a)
    print(f"Mask: {src.width}x{src.height}, {px:.0f} m/px\n")

# lat/lon -> map metres
tr = Transformer.from_crs("+proj=longlat +R=1737400 +no_defs",
                          crs.to_string(), always_xy=True)

print(f"{'Crater':<26}{'pixels':>8}{'cand':>6}{'high':>6}{'%cov':>8}")
print("-"*54)
results = {}
for name,(lat,lon,diam) in craters.items():
    x,y = tr.transform(lon, lat)
    row,col = rowcol(transform, x, y)
    # search radius = crater radius + small margin
    R = int((diam/2 + 200) / px)   # crater radius + 200 m margin
    r0,r1 = max(0,row-R), min(mask.shape[0],row+R)
    c0,c1 = max(0,col-R), min(mask.shape[1],col+R)
    sub = mask[r0:r1, c0:c1]
    if sub.size == 0:
        print(f"{name:<26}{'OUTSIDE IMAGE':>28}")
        continue
    yy,xx = np.indices(sub.shape)
    cy,cx = sub.shape[0]//2, sub.shape[1]//2
    circ = ((yy-cy)**2 + (xx-cx)**2) <= R**2
    crater_px = sub[circ]
    cand = int((crater_px==1).sum())
    high = int((crater_px==2).sum())
    total = int(circ.sum())
    pct = 100*(cand+high)/total if total else 0
    results[name] = (cand, high, total, pct, row, col)
    print(f"{name:<26}{total:>8}{cand:>6}{high:>6}{pct:>7.2f}%")

print("\nInterpretation:")
print("These are the craters the paper found ice in. Any candidate")
print("pixels here = our method agrees with published ice locations.")