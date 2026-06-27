# ============================================================
# LunaVault — Rover Traverse with Solar Power Constraint (Step 5) v2
# ============================================================
# Uses corrected CPR-in-PSR ice mask. A* path now considers BOTH:
#   (1) terrain hazard  : slope > 10 deg impassable, steeper = costlier
#   (2) SOLAR POWER      : permanently shadowed (PSR) pixels cost more,
#                          since a solar rover cannot recharge in shadow.
# The rover therefore prefers sunlit routes, dipping into shadow only
# as needed to reach the ice (which itself lies in shadow).
# ============================================================
import os, heapq
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RS
from scipy import ndimage
import matplotlib.pyplot as plt
from pyproj import Transformer

os.makedirs("outputs/figures", exist_ok=True)

MAX_SLOPE_DEG  = 10.0     # strict terrain limit
DOWNSAMPLE     = 4
TARGET_DIST_KM = 5.0
SHADOW_PENALTY = 3.0      # extra cost multiplier for crossing PSR (solar constraint)

ICE_MASK = "data/processed/ice_cpr_psr.tif"

# ============================================================
# 1. Load ice grid (downsampled)
# ============================================================
with rasterio.open(ICE_MASK) as src:
    H0, W0 = src.height, src.width
    H, W = H0 // DOWNSAMPLE, W0 // DOWNSAMPLE
    ice = src.read(1, out_shape=(H, W), resampling=RS.nearest)
    ice_transform = src.transform * src.transform.scale(W0 / W, H0 / H)
    ice_crs = src.crs
    px_m = abs(ice_transform.a)
    print(f"Grid: {ice.shape}, {px_m:.0f} m/px")

ice_present = (ice >= 1).astype(np.uint8)
print(f"Ice pixels: {int(ice_present.sum())}")

# ============================================================
# 2. Reproject slope AND PSR onto the grid
# ============================================================
def reproject_to_grid(path, resamp=Resampling.bilinear):
    out = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ice_transform, dst_crs=ice_crs,
                  resampling=resamp)
    return out

print("Reprojecting slope...")
slope = reproject_to_grid("data/raw/dtm/LDSM_80S_80MPP_ADJ.TIF")

print("Reprojecting PSR (for solar constraint)...")
psr = reproject_to_grid("data/raw/masks/LPSR_80S_20MPP_ADJ.TIF", Resampling.nearest)
psr_bin = (psr >= 0.5)   # True = permanently shadowed = no solar power
print(f"Shadowed pixels on grid: {int(psr_bin.sum()):,}")

# ============================================================
# 3. Largest ice cluster -> goal
# ============================================================
labels, n = ndimage.label(ice_present)
if n == 0:
    raise RuntimeError("No ice clusters.")
sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n+1))
biggest = int(np.argmax(sizes)) + 1
cluster = (labels == biggest)
passable = np.isfinite(slope) & (slope <= MAX_SLOPE_DEG)

reachable_ice = cluster & passable
if reachable_ice.sum() == 0:
    reachable_ice = (ice_present == 1) & passable
if reachable_ice.sum() == 0:
    raise RuntimeError("No passable ice pixels.")
ice_pts = np.argwhere(reachable_ice)
print(f"Largest ice cluster: {int(sizes.max())} px; passable ice goals: {len(ice_pts)}")

# ============================================================
# 4. Landing site ~5 km from ice, on safe SUNLIT ground
# ============================================================
dist_from_ice = ndimage.distance_transform_edt(~cluster) * px_m / 1000.0
# prefer landing in sunlight (not PSR) and flat
ring = passable & (~psr_bin) & (np.abs(dist_from_ice - TARGET_DIST_KM) < 0.5)
if ring.sum() == 0:
    ring = passable & (~psr_bin) & (np.abs(dist_from_ice - TARGET_DIST_KM) < 2.0)
if ring.sum() == 0:
    ring = passable & (np.abs(dist_from_ice - TARGET_DIST_KM) < 2.0)  # fallback any
cand = np.argwhere(ring)
cand_slopes = slope[cand[:,0], cand[:,1]]
start = tuple(int(v) for v in cand[np.argmin(cand_slopes)])

d2 = ((ice_pts[:,0]-start[0])**2 + (ice_pts[:,1]-start[1])**2)
goal = tuple(int(v) for v in ice_pts[np.argmin(d2)])

tr = Transformer.from_crs(ice_crs.to_string(),
                          "+proj=longlat +R=1737400 +no_defs", always_xy=True)
def to_latlon(r, c):
    x = ice_transform.c + c*ice_transform.a + r*ice_transform.b
    y = ice_transform.f + c*ice_transform.d + r*ice_transform.e
    lon, lat = tr.transform(x, y); return lat, lon
slat, slon = to_latlon(*start); glat, glon = to_latlon(*goal)
print(f"Landing site: lat {slat:.2f}, lon {slon:.2f} | "
      f"{'SHADOW' if psr_bin[start] else 'SUNLIT'}")
print(f"Ice target:   lat {glat:.2f}, lon {glon:.2f}")

# ============================================================
# 5. Cost grid: terrain hazard + SOLAR penalty
# ============================================================
# base terrain cost
cost = np.where(passable, 1.0 + (slope / MAX_SLOPE_DEG) * 5.0, np.inf)
# solar power penalty: shadowed pixels cost SHADOW_PENALTY x more
cost = np.where(psr_bin & np.isfinite(cost), cost * SHADOW_PENALTY, cost)
cost[start] = 1.0; cost[goal] = cost[goal] if np.isfinite(cost[goal]) else 1.0
print(f"Passable pixels: {int(passable.sum())}")

# ============================================================
# 6. A* pathfinding
# ============================================================
def astar(cost, start, goal):
    Hh, Ww = cost.shape
    def h(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])
    open_set=[(0.0,start)]; came={}; g={start:0.0}; visited=set()
    nbrs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur==goal: break
        if cur in visited: continue
        visited.add(cur)
        for dr,dc in nbrs:
            nr,nc=cur[0]+dr,cur[1]+dc
            if 0<=nr<Hh and 0<=nc<Ww:
                c=cost[nr,nc]
                if not np.isfinite(c): continue
                step=c*(1.414 if dr and dc else 1.0)
                ng=g[cur]+step
                if (nr,nc) not in g or ng<g[(nr,nc)]:
                    g[(nr,nc)]=ng; came[(nr,nc)]=cur
                    heapq.heappush(open_set,(ng+h((nr,nc),goal),(nr,nc)))
    if goal not in came and goal!=start: return None
    path=[goal]
    while path[-1]!=start: path.append(came[path[-1]])
    return path[::-1]

print("Running A* (terrain + solar)...")
path = astar(cost, start, goal)

if path is None:
    print("No safe path found.")
else:
    pa=np.array(path)
    seg=np.diff(pa,axis=0)
    path_km=(np.hypot(seg[:,0],seg[:,1])*px_m/1000).sum()
    ps=slope[pa[:,0],pa[:,1]]
    straight=np.hypot(goal[0]-start[0],goal[1]-start[1])*px_m/1000
    shadow_steps = int(psr_bin[pa[:,0], pa[:,1]].sum())
    shadow_frac = 100*shadow_steps/len(path)
    print(f"\n===== ROVER PATH (terrain + solar) =====")
    print(f"Path steps: {len(path)}")
    print(f"Path length: {path_km:.2f} km (straight-line: {straight:.2f} km)")
    print(f"Detour factor: {path_km/straight:.2f}x")
    print(f"Max slope on path: {np.nanmax(ps):.1f} deg")
    print(f"Mean slope on path: {np.nanmean(ps):.1f} deg")
    print(f"Path in shadow: {shadow_steps}/{len(path)} steps ({shadow_frac:.0f}%) "
          f"-- rover minimises shadow time for solar power")

# ============================================================
# 7. Figure
# ============================================================
plt.figure(figsize=(10,10))
plt.imshow(np.clip(slope,0,20), cmap="terrain_r")
plt.colorbar(label="slope (deg)", shrink=0.7)
# show shadow as a blue overlay
plt.imshow(np.ma.masked_where(~psr_bin, psr_bin), cmap="Blues", alpha=0.25)
plt.imshow(np.ma.masked_where(ice_present==0, ice_present), cmap="cool", alpha=0.9)
if path is not None:
    plt.plot(pa[:,1], pa[:,0], "r-", linewidth=2.5, label="rover path")
plt.scatter(start[1], start[0], c="lime", s=200, marker="^",
            edgecolors="black", zorder=5, label="landing (sunlit)")
plt.scatter(goal[1], goal[0], c="magenta", s=220, marker="*",
            edgecolors="white", zorder=5, label="target ice")
plt.legend(loc="upper right")
ttl = f"{path_km:.1f} km, {shadow_frac:.0f}% in shadow" if path else "no path"
plt.title(f"Rover traverse: terrain + solar power constraint\n"
          f"slope <= {MAX_SLOPE_DEG} deg, shadow penalty {SHADOW_PENALTY}x ({ttl})")
plt.savefig("outputs/figures/07_rover_path.png", dpi=130, bbox_inches="tight")
print("\nSaved: outputs/figures/07_rover_path.png")