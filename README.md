# LunaVault

> **Lunar south pole water-ice detection and landing-site selection from Chandrayaan-2 DFSAR data.**

LunaVault is a rule-based radar polarimetry pipeline that ingests ISRO Chandrayaan-2 Dual-Frequency SAR (DFSAR) Level-3C data to detect subsurface water-ice at the lunar south pole, estimate its volume, and rank candidate landing sites for future crewed or robotic missions. No machine learning — every decision is grounded in physical radar scattering models.

---

## Table of Contents

- [Background](#background)
- [Pipeline Overview](#pipeline-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Detection Logic](#detection-logic)
- [Scripts Reference](#scripts-reference)
- [Key Results](#key-results)
- [Outputs](#outputs)
- [Configuration and Thresholds](#configuration-and-thresholds)
- [Known Limitations](#known-limitations)
- [Data Requirements](#data-requirements)
- [References](#references)

---

## Background

The core physical signal is the **Circular Polarization Ratio (CPR)**:

> CPR greater than 1 indicates coherent backscatter — the hallmark of volume scattering from ice.  
> However, rough rocky terrain also produces high CPR, so additional polarimetric parameters are used to disambiguate.

LunaVault fuses three DFSAR-derived products (per Putrevu 2023 and Sinha et al. 2026):

| Parameter | Physical Meaning | Ice Signature |
|-----------|-----------------|---------------|
| **CPR** (Circular Polarization Ratio) | Volume vs. surface scattering | CPR > 1 |
| **SERD** (Single-bounce Eigenvalue Relative Difference) | Surface roughness | SERD > 0.75 (smooth) |
| **TRT** (T-Ratio) | Dielectric constant | TRT < 0.5 (low dielectric ~3, like ice) |

Detection v2 (production) further constrains candidates to **Permanently Shadowed Regions (PSRs)** — cold traps where water ice can survive for billions of years.

---

## Pipeline Overview

```
Chandrayaan-2 DFSAR GeoTIFFs
        |
        +-- [Synthetic Track] -----------------------------------------+
        |   make_test_data.py  -> detect_ice.py                        |
        |   (S1-S4 Stokes, CPR+DOP, 400x400 validation rasters)        |
        |                                                               |
        +-- [Real Track] ----------------------------------------------+
            detect_real.py    (CPR-only exploratory)                   |
            detect_fusion.py  (CPR+SERD+TRT, chunked)                  |
            detect_cpr_psr.py (CPR in PSR, v2 production) ------------+
                    |
                    v
             ice_cpr_psr.tif --> ice_volume.py --> Volume Report
                    |
                    +--> psr_restriction.py
                    +--> validate_paper_craters.py
                    +--> landing_site.py  --> Top 5 landing sites
                    +--> rover_path.py   --> Rover traversal map
```

---

## Repository Structure

```
LunaVault/
|-- src/
|   |-- make_test_data.py         # Generate synthetic Stokes rasters (S1-S4)
|   |-- detect_ice.py             # Synthetic CPR+DOP detection (validation/demo)
|   |-- detect_real.py            # Real data: CPR-only exploratory map
|   |-- detect_fusion.py          # Real data: CPR+SERD+TRT fusion (chunked)
|   |-- detect_cpr_psr.py         # [PRODUCTION] CPR>1 restricted to PSRs (v2)
|   |-- psr_restriction.py        # PSR mask utilities
|   |-- validate_paper_craters.py # Validates vs. Sinha 2026 known craters
|   |-- tune_threshold.py         # Threshold sensitivity / enrichment analysis
|   |-- ice_volume.py             # Ice surface area + volume estimation
|   |-- volume.py                 # Monte Carlo volume + tonnage report
|   |-- landing_site.py           # Landing site scoring (slope + ice proximity)
|   |-- rover_path.py             # Rover traversal path planning
|   |-- diagnose.py               # Diagnostic checks on raster data
|   +-- io_utils.py               # Shared raster I/O helpers
|-- outputs/
|   |-- figures/                  # All generated PNGs
|   +-- ice_volume_report.txt     # Human-readable volume estimate
|-- requirements.txt              # Python dependencies (pinned)
|-- config.yaml                   # Placeholder for future config
+-- CLAUDE.md                     # Developer guidance
```

> **Generated at runtime** (gitignored): `data/processed/*.tif`, `data/processed/volume_report.json`, `data/raw/`.

---

## Quick Start

### 1. Clone and set up the environment

```bash
git clone https://github.com/Gaurang774/LunaVault.git
cd LunaVault
python -m venv venv

# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the synthetic demo (no real data needed)

```bash
python src/make_test_data.py    # generates data/raw/dfsar/test_S1..S4.tif
python src/detect_ice.py        # synthetic CPR+DOP detection
```

### 3. Run the production pipeline (requires real DFSAR GeoTIFFs)

> See [Data Requirements](#data-requirements) for how to obtain the DFSAR data.

```bash
# Update the BASE path in detect_cpr_psr.py and detect_fusion.py first!
python src/detect_cpr_psr.py    # CPR>1 in PSR  --> data/processed/ice_cpr_psr.tif
python src/ice_volume.py        # Volume estimate --> outputs/ice_volume_report.txt
python src/landing_site.py      # Top 5 candidate landing sites
python src/rover_path.py        # Rover traversal map
```

All scripts are **run from the repo root**.

---

## Detection Logic

### Detection v1 — Fusion (CPR + SERD + TRT)

| Tier | Rule | Label |
|------|------|-------|
| Candidate ice | CPR > 1.0 AND SERD > 0.75 | Smooth, volume-scattering |
| High-confidence ice | Candidate AND TRT < 0.5 | Also low-dielectric (pure ice) |

Implemented in `src/detect_fusion.py`.

### Detection v2 — CPR in PSR (Production)

Following Sinha et al. (2026), the primary ice indicator is CPR > 1 **within permanently shadowed regions**. SERD/TRT filters are retained as a secondary subset, not hard gates — actual ice-bearing craters F2, F3, S1 are not smooth.

| Tier | Rule | Label |
|------|------|-------|
| Ice candidate | CPR > 1.0 AND in PSR | 1 |
| Secondary subset | Above AND SERD > 0.6 AND TRT < 0.5 | 2 |

Implemented in `src/detect_cpr_psr.py`.

### Synthetic Demo (CPR + DOP)

Stokes parameters: `CPR = (S1-S4)/(S1+S4)`, `DOP = sqrt(S2^2+S3^2+S4^2)/S1`

Threshold: CPR > 1.0 AND DOP < 0.13, then cleaned with `remove_small_objects` + `binary_closing`.

---

## Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `make_test_data.py` | Generate synthetic Stokes rasters | None | `data/raw/dfsar/test_S*.tif` |
| `detect_ice.py` | Synthetic CPR+DOP detection | Stokes TIFs | `data/processed/ice_mask.tif` |
| `detect_real.py` | Exploratory CPR-only map | CPR GeoTIFF | figure, candidates |
| `detect_fusion.py` | CPR+SERD+TRT fusion (chunked) | CPR/SRD/TRT GeoTIFFs | `real_ice_mask.tif` |
| `detect_cpr_psr.py` | **Production** CPR in PSR | CPR/SRD/TRT + PSR mask | `ice_cpr_psr.tif` |
| `validate_paper_craters.py` | Validate against Sinha 2026 | ice mask | console report |
| `tune_threshold.py` | Threshold sensitivity analysis | CPR/SRD/TRT | enrichment figure |
| `ice_volume.py` | Ice surface area + volume | `ice_cpr_psr.tif` | `ice_volume_report.txt` |
| `volume.py` | Monte Carlo tonnage (P10/P50/P90) | `real_ice_mask.tif` | `volume_report.json` |
| `landing_site.py` | Landing site scoring | ice mask + DEM + slope | top-5 sites + figure |
| `rover_path.py` | Rover traversal path planning | ice mask + DEM | traversal map figure |
| `diagnose.py` | Raster diagnostics | any TIF | console stats |
| `io_utils.py` | Shared I/O helpers | None | None |

---

## Key Results

Based on Chandrayaan-2 DFSAR South Pole mosaic (processed June 2025):

```
Detection: CPR>1 inside Permanently Shadowed Regions
Detected ice pixels:    41,422
Pixel size:             25.0 m  (625 m^2 per pixel)
Ice surface area:       25.89 km^2
```

### Monte Carlo Ice Volume Estimate

| Scenario | Ice Volume (m3) | Ice Mass (tonnes) |
|----------|----------------|-------------------|
| P10 — conservative (2% ice fraction) | 2,588,875 | 2,381,765 |
| P50 — nominal (10% ice fraction) | 12,944,375 | 11,908,825 |
| P90 — optimistic (30% ice fraction) | 38,833,125 | 35,726,475 |

*Depth assumption: top 5 m; ice density: 920 kg/m3*

---

## Outputs

All generated figures are saved to `outputs/figures/`:

| File | Description |
|------|-------------|
| `02_ice_detection.png` | Synthetic CPR+DOP ice detection map |
| `03_real_cpr.png` | Real CPR downsampled overview |
| `04_fusion_ice.png` | Fusion detection (candidate=cyan, high-conf=red) |
| `05_faustini_validation.png` | Validation vs. Faustini crater |
| `06_landing_sites.png` | Top 5 landing site candidates (stars) near ice (cyan) |
| `07_rover_path.png` | Rover traversal path map |
| `08_threshold_enrichment.png` | CPR threshold sensitivity / enrichment analysis |
| `09_cpr_psr_detection.png` | CPR>1 in PSR detection map (v2) |

---

## Configuration and Thresholds

Detection thresholds are module-level constants at the top of each script. **Edit constants to tune, not the logic.**

| Constant | Default | Meaning |
|----------|---------|---------|
| `CPR_T` | `1.0` | Volume scattering threshold |
| `SERD_T` | `0.75` | Surface smoothness (rejects rough rock) |
| `TRT_T` | `0.5` | Low dielectric constant (ice-like) |
| `MIN_SIZE` | varies | Minimum cluster size in pixels |
| `MAX_SLOPE_DEG` | `15.0` | Lander tip-over limit (degrees) |
| `MAX_ICE_DIST_KM` | `10.0` | Max rover distance to ice (km) |

> **Note**: `detect_fusion.py` and `detect_cpr_psr.py` hard-code a `BASE` path to a local DFSAR download. **Update this path before running on a new machine.**

---

## Known Limitations

- **Hardcoded paths**: `BASE` constant in `detect_real.py`, `detect_fusion.py`, and `detect_cpr_psr.py` must be updated per machine/dataset.
- **CPR ambiguity**: CPR > 1 alone cannot distinguish ice from rough terrain. The PSR constraint (v2) significantly reduces false positives but does not eliminate them entirely.
- **No test suite**: scripts are standalone and run manually in sequence.
- **`detect_ice.py`** redefines its own I/O helpers rather than importing from `io_utils.py`.
- **Large rasters**: never `imshow` full-res data — scripts always downsample before plotting.

---

## Data Requirements

| Dataset | Source | Usage |
|---------|--------|-------|
| Chandrayaan-2 DFSAR Level-3C GeoTIFFs (CPR, SERD, TRT) | [ISRO PRADAN](https://pradan.nrsc.gov.in/) | Ice detection |
| LOLA DEM (LDEM_80S_80MPP_ADJ.TIF) | [NASA PDS](https://pds.nasa.gov/) | Landing site terrain |
| LOLA Slope (LDSM_80S_80MPP_ADJ.TIF) | [NASA PDS](https://pds.nasa.gov/) | Slope safety filter |
| PSR mask (LPSR_80S_20MPP_ADJ.TIF) | [NASA PDS / LOLA](https://pds.nasa.gov/) | Cold-trap constraint |

---

## References

- Putrevu, D. et al. (2023). *Chandrayaan-2 DFSAR: Polarimetric analysis of the lunar south pole.* ISRO.
- Sinha, R. K. et al. (2026). *CPR-based water-ice detection in permanently shadowed craters.*
- ISRO DFSAR User Guide (Level-3C polar mosaic product).
- NASA LOLA Planetary Data System — Lunar Digital Elevation Models.

---

*Built with Python 3.14 · rasterio · numpy · scikit-image · geopandas · matplotlib*
