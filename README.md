# manet-heatmap

A Streamlit-based MANET RF coverage and throughput heatmap tool for urban and terrain-aware analysis.

The current codebase is centered on the V23 app. It models coverage on real terrain using DSM data, supports multiple propagation modes, applies optional building attenuation, renders a map overlay, and exports results as GeoTIFF and CSV.

## What it does

This project provides an interactive engineering tool to explore how radio settings, terrain, and buildings affect MANET coverage and achievable throughput over a selected area of interest.

The main workflow is:

- geocode a city
- define an AOI radius
- load or fetch DSM data
- generate a simulation grid
- calculate path loss, SINR, and throughput
- render a heatmap overlay
- export results

## Current capabilities

- Streamlit UI for city selection, AOI radius, grid resolution, radio parameters, antenna geometry, and diagnostics
- Propagation options including:
  - FSPL
  - Two-Ray interference
  - Two-Ray Fresnel-ground
  - Terrain line-of-sight plus diffraction
- Optional building attenuation using rasterized OSM building footprints and line-of-sight raycast masking
- DSM handling through OpenTopography COP30 with local reuse of downloaded terrain data
- Interactive Folium map output
- Export of GeoTIFF and CSV results
- Logging and telemetry for OSM, rasterization, raycast, and I/O behavior

## Technical basis

The RF computation path includes:

- Free Space Path Loss
- noise floor calculation
- SINR to spectral-efficiency mapping
- throughput estimation
- two-ray models
- Fresnel-ground reflection handling
- terrain line-of-sight and knife-edge diffraction
- optional building loss penalties

The code also includes worker-thread time limits, OSM mirror retry behavior, session caching, and diagnostics hooks for long-running tasks.

## Stack

The current app uses:

- Python 3.11
- Streamlit
- Folium
- streamlit-folium
- GeoPandas
- NumPy
- OSMnx
- pandas
- pyproj
- rasterio
- requests

The project configuration also includes formatting and static-analysis settings for:

- Ruff
- isort
- mypy

## Getting started

Create a Python 3.11 environment and install the required dependencies.

Example:

```bash
pip install streamlit folium streamlit-folium geopandas numpy osmnx pandas pyproj rasterio requests branca shapely

```
Run the app with:
```bash
streamlit run manet_heatmap_app.py
```

Known issues:

The current debug log shows an OSM building-fetch failure path under osmnx=2.0.6, including a features_from_bbox() signature mismatch and a downstream KeyError: 'timeout' during Overpass handling.

This means the repository should be treated as an active engineering workbench, not a finished release.

Status:

Active development.
