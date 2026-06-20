# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LunaVault detects water-ice on the lunar south pole from Chandrayaan-2 dual-frequency SAR (DFSAR) data and estimates ice volume. Detection is rule-based on radar polarimetry — no ML. The core physical signal is **CPR (Circular Polarization Ratio)**: CPR > 1 indicates the coherent backscatter expected from ice (but also from rough terrain, which the fusion/DOP stages try to reject).

## Environment & Commands

```bash
source venv/bin/activate          # Python 3.14 venv at ./venv
pip install -r requirements.txt   # geospatial stack: rasterio, numpy, scikit-image, geopandas, matplotlib

python src/make_test_data.py      # generate synthetic Stokes rasters -> data/raw/dfsar/test_S1..S4.tif
python src/detect_ice.py          # synthetic CPR+DOP detection (run on test data)
python src/detect_real.py         # CPR-only candidate map from a real CPR GeoTIFF
python src/detect_fusion.py       # chunked CPR+SRD+TRT fusion on real data -> real_ice_mask.tif
python src/volume.py              # Monte Carlo ice volume from real_ice_mask.tif -> volume_report.json
```

Scripts are standalone (no test suite, no CLI args). Run each from the repo root — paths like `data/processed` are relative to cwd, and each script `os.makedirs(..., exist_ok=True)` its outputs.

## Pipeline & Data Flow

The pipeline has two parallel tracks that converge on the volume estimate:

1. **Synthetic track** (`make_test_data.py` → `detect_ice.py`): generates Stokes parameters S1–S4, derives CPR = (S1−S4)/(S1+S4) and DOP = √(S2²+S3²+S4²)/S1, thresholds `CPR > 1.0 & DOP < 0.13`, then cleans with `remove_small_objects` + `binary_closing`. This is the validation/demo path on small (400×400) rasters.

2. **Real track** (`detect_real.py` → `detect_fusion.py`): operates on large (~12k×12k) Chandrayaan-2 GeoTIFFs. `detect_real.py` is exploratory (CPR-only candidates + downsampled figure). `detect_fusion.py` is the production detector — it fuses three derived products **CPR > 1.0 & SRD < 0.3 & TRT > 1.0** (per Putrevu 2023) and reads the rasters in row-chunks (`CHUNK = 1000`) to stay within memory, writing `data/processed/real_ice_mask.tif`.

3. **Volume** (`volume.py`): reads the ice mask, computes area from pixel geotransform, then Monte Carlo samples ice depth (0–5 m) and ice fraction (2–10%) with density 917 kg/m³ to report P10/P50/P90 tonnage.

`data/processed/` (`cpr.tif`, `dop.tif`, `ice_mask.tif`, `real_ice_mask.tif`, `volume_report.json`) and `outputs/figures/` are generated artifacts. `data/raw/` (except synthetic dfsar) is gitignored, as are `*.tif`/`*.img`.

## Conventions & Gotchas

- **Hardcoded absolute paths**: `detect_real.py` and `detect_fusion.py` point at real DFSAR files under `/home/gaurang/Downloads/ch2_sar_ndxl_.../data/derived/20250630/` (the `BASE` / `CPR_PATH` constants). These must be updated for a different machine or dataset.
- **Detection thresholds** are module-level constants at the top of each script (`CPR_T`, `DOP_T`, `SRD_T`, `TRT_T`, `MIN_SIZE`). Tune detection by editing those, not the logic. Note the fusion source comment uses different thresholds than the constants — trust the constants.
- **`io_utils.py`** holds shared `load_raster` / `load_stokes` / `save_raster` helpers, but `detect_ice.py` redefines its own copies inline rather than importing them.
- **Large-raster rule**: never `imshow` a full-res real raster — downsample (`[::step]` or rasterio `out_shape`) before plotting, as the existing scripts do.
- `report/`, `dashboard/`, `config.yaml`, and `README.md` exist but are currently empty placeholders.
