# ============================================================
# LunaVault — Rover Traverse Path (Official Step 5)
# Demonstrates terrain-aware A* navigation from a safe landing
# zone to an ice deposit ~5 km away, routing around steep craters.
# ============================================================
# Mission rationale: landers avoid touching down directly on rough
# ice-bearing crater terrain; they land on safe flat ground nearby
# and the rover drives in. We place the landing site ~5 km from the
# largest ice cluster and find the safest driveable route.
# Rule: slopes > 10 deg impassable (STRICT); flatter = cheaper.
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

MAX_SLOPE_DEG = 10.0      # STRICT
DOWNSAMPLE    = 4
TARGET_DIST_KM = 5.0      # desired landing distance from ice cluster

# ============================================================
# 1. Load ice grid (downsampled)
# ============================================================
with rasterio.open("data/processed/ice_cpr_psr.tif") as src:
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
# 2. Reproject slope onto the grid
# ============================================================
slope = np.full((H, W), np.nan, dtype=np.float32)
with rasterio.open("data/raw/dtm/LDSM_80S_80MPP_ADJ.TIF") as src:
    reproject(source=rasterio.band(src, 1), destination=slope,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=ice_transform, dst_crs=ice_crs,
              resampling=Resampling.bilinear)

# ============================================================
# 3. Find the LARGEST ice cluster (the main deposit)
# ============================================================
labels, n = ndimage.label(ice_present)
if n == 0:
    raise RuntimeError("No ice clusters found.")
sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n+1))
biggest = int(np.argmax(sizes)) + 1
cluster = (labels == biggest)
print(f"Largest ice cluster: label {biggest}, {int(sizes.max())} px")

passable = np.isfinite(slope) & (slope <= MAX_SLOPE_DEG)

# Goal = a REACHABLE ice pixel: an ice pixel that is itself passable
# (flat enough), preferring ones near the cluster edge, not buried in
# a steep crater interior.
reachable_ice = cluster & passable
if reachable_ice.sum() == 0:
    print("WARNING: no passable ice pixels in largest cluster; "
          "using any passable ice pixel.")
    reachable_ice = (ice_present == 1) & passable
if reachable_ice.sum() == 0:
    raise RuntimeError("No passable ice pixels anywhere at slope <= "
                       f"{MAX_SLOPE_DEG} deg.")
ice_pts = np.argwhere(reachable_ice)
print(f"Passable ice pixels available as goal: {len(ice_pts)}")

# ============================================================
# 4. Pick a safe landing site ~5 km from the cluster
# ============================================================
dist_from_ice = ndimage.distance_transform_edt(~cluster) * px_m / 1000.0

ring = passable & (np.abs(dist_from_ice - TARGET_DIST_KM) < 0.5)
if ring.sum() == 0:
    ring = passable & (np.abs(dist_from_ice - TARGET_DIST_KM) < 2.0)
cand = np.argwhere(ring)
cand_slopes = slope[cand[:,0], cand[:,1]]
start = tuple(int(v) for v in cand[np.argmin(cand_slopes)])

# Goal = the passable ice pixel CLOSEST to the chosen start
d2 = ((ice_pts[:,0]-start[0])**2 + (ice_pts[:,1]-start[1])**2)
goal = tuple(int(v) for v in ice_pts[np.argmin(d2)])

print(f"Landing site pixel: {start} | {dist_from_ice[start]:.1f} km from ice | "
      f"slope {slope[start]:.1f} deg")
print(f"Goal (reachable ice) pixel: {goal} | slope {slope[goal]:.1f} deg")

# lat/lon
tr = Transformer.from_crs(ice_crs.to_string(),
                          "+proj=longlat +R=1737400 +no_defs", always_xy=True)
def to_latlon(r, c):
    x = ice_transform.c + c*ice_transform.a + r*ice_transform.b
    y = ice_transform.f + c*ice_transform.d + r*ice_transform.e
    lon, lat = tr.transform(x, y); return lat, lon
slat, slon = to_latlon(*start)
glat, glon = to_latlon(*goal)
print(f"Landing site: lat {slat:.2f}, lon {slon:.2f}")
print(f"Ice target:   lat {glat:.2f}, lon {glon:.2f}")
# lat/lon of the chosen landing site
tr = Transformer.from_crs(ice_crs.to_string(),
                          "+proj=longlat +R=1737400 +no_defs", always_xy=True)
def to_latlon(r, c):
    x = ice_transform.c + c*ice_transform.a + r*ice_transform.b
    y = ice_transform.f + c*ice_transform.d + r*ice_transform.e
    lon, lat = tr.transform(x, y); return lat, lon
slat, slon = to_latlon(*start)
glat, glon = to_latlon(*goal)
print(f"Landing site: lat {slat:.2f}, lon {slon:.2f}")
print(f"Ice target:   lat {glat:.2f}, lon {glon:.2f}")

# ============================================================
# 5. Cost grid: flat cheap, steep expensive, >10deg = wall
# ============================================================
cost = np.where(passable, 1.0 + (slope / MAX_SLOPE_DEG) * 5.0, np.inf)
cost[start] = 1.0; cost[goal] = 1.0
print(f"Passable pixels: {int(passable.sum())}")

# ============================================================
# 6. A* pathfinding
# ============================================================
def astar(cost, start, goal):
    Hh, Ww = cost.shape
    def h(a,b): return np.hypot(a[0]-b[0], a[1]-b[1])
    open_set = [(0.0, start)]; came={}; g={start:0.0}; visited=set()
    nbrs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal: break
        if cur in visited: continue
        visited.add(cur)
        for dr,dc in nbrs:
            nr,nc = cur[0]+dr, cur[1]+dc
            if 0<=nr<Hh and 0<=nc<Ww:
                c = cost[nr,nc]
                if not np.isfinite(c): continue
                step = c*(1.414 if dr and dc else 1.0)
                ng = g[cur]+step
                if (nr,nc) not in g or ng < g[(nr,nc)]:
                    g[(nr,nc)]=ng; came[(nr,nc)]=cur
                    heapq.heappush(open_set,(ng+h((nr,nc),goal),(nr,nc)))
    if goal not in came and goal!=start: return None
    path=[goal]
    while path[-1]!=start: path.append(came[path[-1]])
    return path[::-1]

print("Running A* pathfinding...")
path = astar(cost, start, goal)

if path is None:
    print("No safe path found. Try relaxing MAX_SLOPE_DEG.")
else:
    pa = np.array(path)
    seg = np.diff(pa, axis=0)
    path_km = (np.hypot(seg[:,0], seg[:,1]) * px_m / 1000).sum()
    ps = slope[pa[:,0], pa[:,1]]
    straight = np.hypot(goal[0]-start[0], goal[1]-start[1]) * px_m / 1000
    print(f"\n===== ROVER PATH =====")
    print(f"Path steps: {len(path)}")
    print(f"Path length: {path_km:.2f} km (straight-line: {straight:.2f} km)")
    print(f"Detour factor: {path_km/straight:.2f}x")
    print(f"Max slope on path: {np.nanmax(ps):.1f} deg")
    print(f"Mean slope on path: {np.nanmean(ps):.1f} deg")

    # --- Push results to the dashboard ---
    import dash_io
    dash_io.update("rover", {
        "path_km": round(float(path_km), 2),
        "straight_km": round(float(straight), 2),
        "detour": round(float(path_km / straight), 2),
        "max_slope_deg": round(float(np.nanmax(ps)), 1),
        "limit_slope_deg": MAX_SLOPE_DEG,
    })

# ============================================================
# 7. Figure
# ============================================================
plt.figure(figsize=(10,10))
plt.imshow(np.clip(slope,0,20), cmap="terrain_r")
plt.colorbar(label="slope (deg)", shrink=0.7)
plt.imshow(np.ma.masked_where(ice_present==0, ice_present), cmap="cool", alpha=0.9)

if path is not None:
    pa = np.array(path)
    plt.plot(pa[:,1], pa[:,0], "r-", linewidth=2.5, label="rover path")
    rs = pa[:,0]; cs = pa[:,1]
    pad = 80
    plt.xlim(cs.min()-pad, cs.max()+pad)
    plt.ylim(rs.max()+pad, rs.min()-pad)  # inverted y
    title_path = f"{path_km:.1f} km path"
else:
    title_path = "NO SAFE PATH (ice in steep crater)"

plt.scatter(start[1], start[0], c="lime", s=200, marker="^",
            edgecolors="black", zorder=5, label="landing site")
plt.scatter(goal[1], goal[0], c="magenta", s=220, marker="*",
            edgecolors="white", zorder=5, label="target ice")
plt.legend(loc="upper right")
plt.title(f"Rover traverse: landing -> ice deposit\n"
          f"strict (slope <= {MAX_SLOPE_DEG} deg), {title_path}")
plt.savefig("outputs/figures/07_rover_path.png", dpi=130, bbox_inches="tight")
print("\nSaved: outputs/figures/07_rover_path.png")