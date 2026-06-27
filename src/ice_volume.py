# ============================================================
# LunaVault — Ice Volume Estimate v2 (Official Step 6)
# Using corrected CPR-in-PSR detection (validated vs paper craters)
# ============================================================
import os, numpy as np, rasterio

os.makedirs("outputs", exist_ok=True)

# CORRECTED mask: CPR>1 inside PSRs (validated against Sinha 2026 craters)
with rasterio.open("data/processed/ice_cpr_psr.tif") as src:
    ice = src.read(1)
    px_m = abs(src.transform.a)
print(f"Pixel size: {px_m:.1f} m  ->  pixel area: {px_m**2:.0f} m^2")

n_ice = int((ice >= 1).sum())          # all detected ice (1 and 2)
print(f"Detected ice pixels (CPR>1 in PSR): {n_ice:,}")

pixel_area_m2 = px_m ** 2
area_total_m2 = n_ice * pixel_area_m2
area_total_km2 = area_total_m2 / 1e6
print(f"Detected ice surface area: {area_total_km2:.2f} km^2")

DEPTH_M = 5.0
bulk_volume_m3 = area_total_m2 * DEPTH_M
print(f"Bulk near-surface volume (area x 5 m): {bulk_volume_m3/1e9:.4f} km^3")

ICE_DENSITY = 920.0
scenarios = {
    "P10 (conservative, 2% ice)":  0.02,
    "P50 (nominal, 10% ice)":      0.10,
    "P90 (optimistic, 30% ice)":   0.30,
}

print("\n===== ICE VOLUME ESTIMATE (top 5 m) =====")
print(f"{'Scenario':<32}{'Ice vol (m^3)':>16}{'Ice mass (t)':>16}{'Water (ML)':>14}")
results = []
for name, frac in scenarios.items():
    ice_vol_m3 = bulk_volume_m3 * frac
    ice_mass_t = ice_vol_m3 * ICE_DENSITY / 1000
    water_ML = ice_vol_m3 * ICE_DENSITY / 1e6   # million litres
    results.append((name, frac, ice_vol_m3, ice_mass_t, water_ML))
    print(f"{name:<32}{ice_vol_m3:>16,.0f}{ice_mass_t:>16,.0f}{water_ML:>14,.0f}")

p50 = results[1]
print(f"\nNominal (P50) interpretation:")
print(f"  ~{p50[2]/1e6:.2f} million m^3 of ice")
print(f"  ~{p50[3]/1e6:.2f} million tonnes of ice")
print(f"  ~{p50[4]:.0f} million litres of water equivalent")

with open("outputs/ice_volume_report.txt", "w") as f:
    f.write("LunaVault - Ice Volume Estimate v2 (corrected CPR-in-PSR)\n")
    f.write("="*58 + "\n\n")
    f.write(f"Detection: CPR>1 inside PSRs, validated vs Sinha 2026 craters\n")
    f.write(f"Detected ice pixels: {n_ice:,}\n")
    f.write(f"Pixel size: {px_m:.1f} m  (area {pixel_area_m2:.0f} m^2)\n")
    f.write(f"Ice surface area: {area_total_km2:.2f} km^2\n")
    f.write(f"Assumed depth: top {DEPTH_M:.0f} m\n")
    f.write(f"Bulk volume: {bulk_volume_m3/1e9:.4f} km^3\n")
    f.write(f"Ice density: {ICE_DENSITY:.0f} kg/m^3\n\n")
    f.write(f"{'Scenario':<32}{'Ice vol (m^3)':>16}{'Ice mass (t)':>16}\n")
    for name, frac, v, m, w in results:
        f.write(f"{name:<32}{v:>16,.0f}{m:>16,.0f}\n")
print("\nSaved: outputs/ice_volume_report.txt")