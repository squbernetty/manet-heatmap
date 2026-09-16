# +=====================================================================+
# | MANET Heatmap App (V23) — DSM local-first OSM, failover & telemetry |
# +=====================================================================+
# |
# | What this app does:
# | - Interactive tool to model MANET coverage and throughput on real terrain (DSM).|
# | - Renders a heat layer on a map and exports GeoTIFF and CSV.                    |
# | ------------------------------------------------------------------------------- |
# | Core flow:
# | - City → AOI radius → DSM load/fetch → Grid build                               |
# | - Path loss and SINR → Throughput → Color mapping → Map overlay → Export        |
# | ------------------------------------------------------------------------------------- |
# | RF / Propagation:
# | - FSPL baseline with user RF inputs: frequency, bandwidth, Tx power, Tx/Rx gains, NF. |
# | - Optional building attenuation via rasterized footprints and raycast masking.        |
# | -------------------------------------------------------------------------------       |
# | OSM buildings (local-first + failover):
# | - Load from ./data/osm/<city>/ (.gpkg/.geojson) if present.                        |
# | - Else query Overpass with mirror rotation and per-call timeout.                   |
# | - UI wait cap prevents blocking the main thread.                                   |
# | - Auto-shrink AOI on failure; cache only non-empty results.                        |
# | ------------------------------------------------------------------------------- |
# | DSM data:
# | - COP30 from OpenTopography with API key (sidebar).                                 |
# | - Stored under ./data/dsm/<city>/ and reused.                                       |
# | ------------------------------------------------------------------------------- |
# | Outputs:
# | - GeoTIFF and CSV saved to ./data/outputs/<city>/.                                   |
# | - Folium map shown in the app.                                                       |
# | ------------------------------------------------------------------------------- |
# | Sidebar controls:
# | - City, AOI radius, grid step.                                                       |
# | - RF: freq, BW, Tx power, Tx/Rx gains, NF.                                           |
# | - Targets: Mbps threshold / spectral efficiency.                                     |
# | - OSM: Overpass timeout, UI wait cap, Max AOI (buildings), auto-shrink, skip-if-slow.|
# | - Diagnostics: logging level, reset/unlock.                                          |
# | ------------------------------------------------------------------------------- |
# | Reliability & telemetry:
# | - Long calls run in a worker thread with time limits.                                |
# | - Overpass retries across mirrors.                                                   |
# | - Telemetry to debugging.txt; session cache reduces repeat work.                     |
# +---------------------------------------------------------------------+

from __future__ import annotations

# ---- std libraries ----
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

# ---- 3rd-party libraries ----
import folium
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import rasterio
import requests
import streamlit as st
from osmnx import projection as oxproj
from rasterio.features import rasterize
from rasterio.transform import from_origin
from streamlit_folium import st_folium

# ------------------- App setup -------------------
st.set_page_config(page_title="MANET Heatmap Simulation", layout="wide")

APP_NAME = "manet_heatmap"
LOG_FILE = Path(__file__).with_name("debugging.txt")
# Initialize state
_defaults: dict[str, Any] = {
    "run_sim": False,
    "_sim_running": False,
    "res": None,
    "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
    "_launched_once": False,
    "_last_los_mask": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

SESSION_ID: str = str(st.session_state["session_id"])

# --- Telemetry container & helpers ---
st.session_state.setdefault("_telemetry", {"osm": {}, "rasterize": {}, "raycast": {}, "io": {}})


def _twrite(section: str, **kv):
    d = st.session_state["_telemetry"].setdefault(section, {})
    d.update(kv)


def _tappend(section: str, key: str, value):
    d = st.session_state["_telemetry"].setdefault(section, {})
    a = d.get(key, [])
    if not isinstance(a, list):
        a = [a]
    a.append(value)
    d[key] = a


def _safe_twrite(section: str, **kv) -> None:
    try:
        _twrite(section, **kv)
    except Exception:
        pass


# ------------------- Logging -------------------
def init_logging() -> logging.Logger:
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, (logging.FileHandler, logging.StreamHandler)):
            root.removeHandler(h)

    mode = "a" if st.session_state.get("_launched_once") else "w"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(LOG_FILE, mode=mode, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)
    root.setLevel(logging.INFO)

    for noisy in ("rasterio", "fiona", "shapely", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger_ = logging.getLogger(APP_NAME)
    if not st.session_state.get("_launched_once"):
        logger_.info("=== MANET Heatmap App Launched ===")
        logger_.info("Session ID: %s", SESSION_ID)
        st.session_state["_launched_once"] = True
    logger_.info("Logging to %s (mode=%s)", LOG_FILE, mode)
    return logger_


# ------------------- OSMnx settings -------------------
def configure_osmnx(logger: logging.Logger) -> None:
    base_dir = Path(__file__).resolve().parent
    cache_dir = base_dir / "data" / "osmnx_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.log_console = False

    # Start from existing kwargs while avoiding duplicate request arguments.
    rk = dict(getattr(ox.settings, "requests_kwargs", {}) or {})

    # OSMnx supplies timeout=settings.requests_timeout explicitly.
    # Keeping timeout in requests_kwargs would pass it twice to requests.
    rk.pop("timeout", None)
    rk.pop("headers", None)

    ox.settings.requests_kwargs = rk

    # Set timeout via the dedicated attribute (not via requests_kwargs)
    try:
        ox.settings.requests_timeout = (120, 120)  # connect, read
    except Exception:
        ox.settings.requests_timeout = 120

    # Set User-Agent via osmnx's own setting if available
    try:
        ox.settings.default_user_agent = f"{APP_NAME}/{SESSION_ID}"
    except Exception:
        # Older versions may not have default_user_agent; then just leave default
        pass

    if hasattr(ox.settings, "max_query_area_size"):
        ox.settings.max_query_area_size = 250_000_000
    if hasattr(ox.settings, "overpass_max_area"):
        ox.settings.overpass_max_area = 250_000_000
    # ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"  # optional

    logger.info(
        "osmnx=%s cache=%s timeout=%s user_agent=%s",
        getattr(ox, "__version__", "unknown"),
        ox.settings.cache_folder,
        getattr(ox.settings, "requests_timeout", None),
        getattr(ox.settings, "default_user_agent", None),
    )


def configure_osmnx_once(logger: logging.Logger) -> None:
    if st.session_state.get("_osmnx_cfg"):
        return
    configure_osmnx(logger)
    st.session_state["_osmnx_cfg"] = True


logger = init_logging()
configure_osmnx_once(logger)  # call AFTER the function is defined


# ------------------- Telemetry helpers -------------------
def log_resources(label: str) -> None:
    try:
        try:
            import psutil as psutil_mod
        except Exception:
            psutil_mod = None

        cpu_s = ""
        mem_s = ""
        if psutil_mod:
            cpu = psutil_mod.cpu_percent(interval=None)
            vmem = psutil_mod.virtual_memory()
            rss_mb = psutil_mod.Process(os.getpid()).memory_info().rss / 1e6
            cpu_s = f"CPU={cpu:.1f}%"
            mem_s = f"RAM={vmem.percent:.1f}% RSS={rss_mb:.1f}MB"

        gpu_s = "no-GPU"
        try:
            import pynvml

            try:
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpu_s = (
                    f"GPU util={util.gpu}% mem={mem.used/1048576:.0f}/{mem.total/1048576:.0f}MiB"
                )
            except Exception:
                pass
        except Exception:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    gpu_s = f"GPU0 load={g.load:.2f} mem={g.memoryUtil:.2f}"
            except Exception:
                gpu_s = "no-GPU"

        logger.info("[RES %s] %s %s %s", label, cpu_s, mem_s, gpu_s)
    except Exception:
        logger.exception("log_resources failed")


@contextmanager
def time_block(tag: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("TIMER[%s] %.3fs", tag, dt)


def log_run_context(tag: str, params: Mapping[str, Any]) -> None:
    try:
        import platform

        logger.info(
            "[%s] Python=%s NumPy=%s pandas=%s rasterio=%s pyproj=%s",
            tag,
            platform.python_version(),
            np.__version__,
            pd.__version__,
            getattr(rasterio, "__version__", "?"),
            getattr(pyproj, "__version__", "?"),
        )
        for k, v in sorted(params.items()):
            logger.info("[%s] %s=%r", tag, k, v)
    except Exception:
        logger.exception("log_run_context failed")


def dump_arrays(run_id: str, out_dir: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {}
        for name, arr in arrays.items():
            fname = out_dir / f"{run_id}_{name}.npy"
            np.save(fname, arr)
            manifest[name] = {
                "file": str(fname),
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
            }
        with open(out_dir / f"{run_id}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Saved debug arrays to %s", out_dir)
    except Exception:
        logger.exception("dump_arrays failed")


def sanity_check_physics(tag: str, freq_mhz: float, d_m: np.ndarray, pl_db: np.ndarray) -> None:
    try:
        if float(np.nanmin(pl_db)) < 0:
            logger.warning("[%s] Path loss < 0 dB (min=%.2f dB).", tag, float(np.nanmin(pl_db)))
        center = pl_db.shape[0] // 2
        row = np.asarray(pl_db[center, :], dtype=float)
        drow = np.asarray(d_m[center, :], dtype=float)
        order = np.argsort(drow)
        diffs = np.diff(row[order])
        if np.any(diffs < -1.0):
            logger.warning("[%s] Non-monotonic loss >1 dB in center row.", tag)
        logger.info(
            "[%s] PL stats: min=%.2f max=%.2f p50=%.2f p95=%.2f",
            tag,
            float(np.nanmin(pl_db)),
            float(np.nanmax(pl_db)),
            float(np.nanpercentile(pl_db, 50)),
            float(np.nanpercentile(pl_db, 95)),
        )
    except Exception:
        logger.exception("sanity_check_physics failed")


# ------------------- Caches -------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_geocode_city(name: str) -> tuple[float, float]:
    lat, lon = ox.geocode(name)
    return float(lat), float(lon)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_load_dsm(path: Path) -> tuple[np.ndarray, Any, Any]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.crs, ds.bounds


@st.cache_data(ttl=86400, show_spinner=False)
def cached_fetch_buildings(
    bbox: tuple[float, float, float, float],
    freshness_bucket: int,
) -> gpd.GeoDataFrame:
    lat_s, lat_n, lon_w, lon_e = bbox
    return fetch_osm_buildings_bbox(lat_s, lat_n, lon_w, lon_e)


# ------------------- Sidebar Inputs -------------------
st.title("MANET Coverage & Throughput Heatmap")
st.write("App loaded at", time.strftime("%H:%M:%S"))

st.sidebar.header("Area of Interest")
city: str = st.sidebar.text_input("City Name", "Tallinn, Estonia")
aoi_radius_km: float = float(st.sidebar.slider("AOI Radius (km)", 1.0, 10.0, 2.5, step=0.5))
grid_resolution_m: int = st.sidebar.slider(
    "Grid resolution (m)",
    min_value=1,
    max_value=100,
    value=int(st.session_state.get("grid_resolution_m", 20)),
    step=1,
    key="grid_resolution_m",
)
st.sidebar.header("Radio Parameters")
frequency_mhz: float = st.sidebar.slider(
    "Center frequency (MHz)",
    min_value=300.0,
    max_value=6000.0,
    value=float(st.session_state.get("frequency_mhz", 2400.0)),
    step=1.0,
    key="frequency_mhz",
)
bandwidth_mhz: int = st.sidebar.slider(
    "Bandwidth (MHz)",
    min_value=1,
    max_value=100,
    value=int(st.session_state.get("bandwidth_mhz", 20)),
    step=1,
    key="bandwidth_mhz",
)
tx_power_dbm: int = int(st.sidebar.slider("TX Power (dBm)", -10, 37, 30))
tx_gain_dbi: int = int(st.sidebar.slider("TX Antenna Gain (dBi)", 0, 15, 5))
rx_gain_dbi: int = int(st.sidebar.slider("RX Antenna Gain (dBi)", 0, 15, 5))
target_mbps: int = st.sidebar.slider(
    "Target Throughput (Mbps)",
    min_value=1,
    max_value=100,
    value=int(st.session_state.get("target_mbps", 10)),
    step=1,
    key="target_mbps",
)
st.sidebar.header("Propagation Model")
model_choice: str = st.sidebar.selectbox(
    "Propagation model",
    [
        "FSPL (Free Space)",
        "Two-Ray (Interference)",
        "Terrain (LoS + diffraction)",
        "Two-Ray (Fresnel ground)",
    ],
)

st.sidebar.divider()
combine_worst: bool = bool(st.sidebar.toggle("Combine models (worst case)", value=False))
combine_models: list[str] = list(
    st.sidebar.multiselect(
        "Models to include",
        [
            "FSPL (Free Space)",
            "Two-Ray (Interference)",
            "Terrain (LoS + diffraction)",
            "Two-Ray (Fresnel ground)",
        ],
        default=["FSPL (Free Space)", "Terrain (LoS + diffraction)"],
        disabled=not combine_worst,
    )
)

# Fresnel specifics
if model_choice == "Two-Ray (Fresnel ground)":
    _sfx = st.session_state["session_id"]
    ground_type: str = str(
        st.sidebar.selectbox(
            "Ground type",
            ["generic", "dry_soil", "wet_soil", "asphalt", "water"],
            index=0,
            help="Affects Fresnel reflection (εr, σ).",
            key=f"fresnel_ground_type_{_sfx}",
        )
    )
    polarization: str = str(
        st.sidebar.selectbox(
            "Polarization",
            ["V", "H"],
            index=0,
            help="V ≈ TM (parallel), H ≈ TE (perpendicular).",
            key=f"fresnel_polarization_{_sfx}",
        )
    )
else:
    ground_type = "generic"
    polarization = "V"

# Antenna geometry
st.sidebar.header("Antenna Geometry")
tx_height_m: int = int(st.sidebar.slider("TX Antenna Height AGL (m)", 1, 50, 10, step=1))
rx_height_m: int = int(st.sidebar.slider("RX Antenna Height AGL (m)", 1, 50, 2, step=1))

st.sidebar.header("Environment")
k_factor: float = float(
    st.sidebar.slider("Refraction k-factor (Earth curvature)", 1.0, 2.0, 1.333, step=0.001)
)

st.sidebar.header("Buildings (OSM)")
apply_bldg: bool = bool(
    st.sidebar.toggle("Apply building attenuation (OSM)", key="apply_bldg_loss", value=False)
)
if apply_bldg:
    for _k in ("building_loss_db", "bldg_nlos_penalty_db"):
        v = st.session_state.get(_k)
        if isinstance(v, int):
            st.session_state.pop(_k, None)

    building_loss_db: float = st.sidebar.slider(
        "Penetration loss on building (dB)",
        min_value=0.0,
        max_value=40.0,
        value=float(st.session_state.get("building_loss_db", 12.0)),
        step=0.5,
        key="building_loss_db",
        help="Applied where the RX cell lies on a building footprint.",
    )

    bldg_nlos_penalty_db: float = st.sidebar.slider(
        "Extra NLOS penalty (dB)",
        min_value=0.0,
        max_value=40.0,
        value=float(st.session_state.get("bldg_nlos_penalty_db", 8.0)),
        step=0.5,
        key="bldg_nlos_penalty_db",
        help="Extra loss when a building blocks TX→RX LoS (raycast).",
    )
else:
    building_loss_db = 0.0
    bldg_nlos_penalty_db = 0.0

enable_bldg_raycast: bool = bool(
    st.sidebar.toggle(
        "Enable building NLOS raycast",
        value=True,
        disabled=not apply_bldg,
        help="Checks if buildings block TX→RX lines; adds the NLOS penalty per blocked cell.",
    )
)

# ------------------- Data roots -------------------
data_root = Path.home() / "manet-heatmap" / "data"
dsm_dir = data_root / "DSM"
dsm_dir.mkdir(parents=True, exist_ok=True)
osm_dir = data_root  # keep at root for simple pathing (can switch to data_root / "OSM")

# OpenTopography key
st.sidebar.text_input(
    "OpenTopography API Key",
    key="ot_key_ui",
    value=os.environ.get("OPENTOPOGRAPHY_API_KEY", ""),
    type="password",
)


def get_api_key() -> str:
    key = st.session_state.get("ot_key_ui") or os.environ.get("OPENTOPOGRAPHY_API_KEY", "")
    key = (key or "").strip().strip("<>").strip('"').strip("'")
    if key:
        os.environ["OPENTOPOGRAPHY_API_KEY"] = key
    return key


# ------------------- Diagnostics UI -------------------
with st.expander("OSM / Buildings debug", expanded=False):
    t = st.session_state.get("_telemetry", {})
    st.json(t)
    try:
        dbg_json = json.dumps(t, indent=2, default=str)
        st.download_button(
            "Download OSM debug JSON",
            data=dbg_json,
            file_name=f"osm_debug_{SESSION_ID}.json",
            mime="application/json",
        )
    except Exception:
        pass

with st.sidebar.expander("Diagnostics", expanded=False):
    debug_logging: bool = st.checkbox("Enable DEBUG logging", key="debug_logging", value=False)
    save_arrays: bool = st.checkbox("Save debug arrays (.npy)", key="save_arrays", value=False)

    st.number_input(
        "Global watchdog (s)",
        min_value=30.0,
        max_value=600.0,
        value=float(st.session_state.get("watchdog_seconds", 240.0)),
        step=30.0,
        key="watchdog_seconds",
    )

    st.caption("OSM / Overpass & Raycast controls")
    osm_timeout_s = st.slider(
        "Overpass timeout (s)",
        min_value=30,
        max_value=300,
        value=int(st.session_state.get("osm_timeout_s", 120)),
        step=5,
        key="osm_timeout_s",
    )

st.sidebar.slider(
    "Buildings cache freshness (days)",
    min_value=0,
    max_value=30,
    value=int(st.session_state.get("buildings_cache_days", 3)),
    step=1,
    key="buildings_cache_days",
)

st.sidebar.slider(
    "Raycast time budget (s)",
    min_value=1.0,
    max_value=120.0,
    value=float(st.session_state.get("raycast_time_budget_s", 8.0)),
    step=1.0,
    key="raycast_time_budget_s",
    help="Hard budget for the building raycast loop.",
)
st.markdown("**Buildings raycast limits**")
st.number_input("Raycast samples", key="bldg_samples", min_value=3, max_value=200, value=24, step=1)
st.number_input(
    "Raycast tile (px)", key="bldg_tile", min_value=64, max_value=1024, value=256, step=32
)
st.number_input("Raycast stride", key="bldg_stride", min_value=1, max_value=8, value=1, step=1)
st.sidebar.slider(
    "Raycast timeout (s)",
    min_value=1,
    max_value=600,
    value=int(st.session_state.get("raycast_timeout_s", 60)),
    step=1,
    key="raycast_timeout_s",
    help="Upper bound for OSM/building tasks.",
)

if st.button("Reset state (clear results & LoS)", type="secondary"):
    st.session_state["res"] = None
    st.session_state["_last_los_mask"] = None
    st.toast("State cleared.")

if st.button("Force unlock Run button"):
    st.session_state["_sim_running"] = False
    st.session_state["run_sim"] = False
    st.toast("Run state unlocked.")

effective = logging.DEBUG if debug_logging else logging.INFO
logging.getLogger().setLevel(effective)
for h in logging.getLogger().handlers:
    h.setLevel(effective)

st.caption(f"Log file: {LOG_FILE}")


# ------------------- Geometry / BBox helpers -------------------
@dataclass(frozen=True)
class BBox:
    lat_s: float
    lat_n: float
    lon_w: float
    lon_e: float


@st.cache_data(show_spinner=False)
def geocode_city_with_buffer_km(
    city: str, radius_km: float
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    lat_f, lon_f = cached_geocode_city(city)
    lat = float(lat_f)
    lon = float(lon_f)

    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * max(0.01, math.cos(math.radians(lat)))

    dlat = float(radius_km) / km_per_deg_lat
    dlon = float(radius_km) / km_per_deg_lon

    lat_s = max(-90.0, lat - dlat)
    lat_n = min(90.0, lat + dlat)
    lon_w = lon - dlon
    lon_e = lon + dlon

    if lon_w < -180.0:
        lon_w += 360.0
    if lon_w > 180.0:
        lon_w -= 360.0
    if lon_e < -180.0:
        lon_e += 360.0
    if lon_e > 180.0:
        lon_e -= 360.0

    lat_s = float(round(lat_s, 6))
    lat_n = float(round(lat_n, 6))
    lon_w = float(round(lon_w, 6))
    lon_e = float(round(lon_e, 6))
    lat = float(round(lat, 6))
    lon = float(round(lon, 6))

    return (lat, lon), (lat_s, lat_n, lon_w, lon_e)


def bbox_to_dsm_path(bbox: tuple[float, float, float, float]) -> Path:
    lat_s, lat_n, lon_w, lon_e = bbox
    name = f"dsm_{lat_s:.4f}_{lat_n:.4f}_{lon_w:.4f}_{lon_e:.4f}.tif"
    return dsm_dir / name


# ------------------- Radio/link helpers -------------------
def fspl_db(distance_m: np.ndarray, freq_mhz: float) -> np.ndarray:
    d_km = np.maximum(distance_m / 1000.0, 1e-6)
    return 32.44 + 20 * np.log10(d_km) + 20 * np.log10(freq_mhz)


def noise_floor_dbm(bw_mhz: float, noise_figure_db: float = 7.0) -> float:
    bw_hz = float(bw_mhz) * 1e6
    return -174 + 10 * math.log10(bw_hz) + float(noise_figure_db)


def spectral_efficiency_from_sinr_db(sinr_db: np.ndarray) -> np.ndarray:
    th = np.array([-5, 0, 3, 6, 10, 15, 20, 25, 30], dtype=float)
    se = np.array([0.05, 0.2, 0.4, 0.8, 1.6, 2.6, 3.5, 4.2, 4.8], dtype=float)
    idx = np.clip(np.searchsorted(th, sinr_db, side="right") - 1, 0, th.size - 1)
    return se[idx]


def wavelength_m(freq_mhz: float) -> float:
    return 3e8 / (float(freq_mhz) * 1e6)


def two_ray_path_loss_db(d_m: np.ndarray, freq_mhz: float, ht_m: float, hr_m: float) -> np.ndarray:
    lam = wavelength_m(freq_mhz)
    d_dir = np.sqrt(np.maximum(d_m, 1e-6) ** 2 + (ht_m - hr_m) ** 2)
    d_refl = np.sqrt(np.maximum(d_m, 1e-6) ** 2 + (ht_m + hr_m) ** 2)
    delta = np.abs(d_refl - d_dir)
    interference_linear = 4.0 * np.sin(np.pi * delta / lam) ** 2
    interference_linear = np.maximum(interference_linear, 1e-6)
    pl_fspl = fspl_db(d_dir, freq_mhz)
    return pl_fspl - 10.0 * np.log10(interference_linear)


def earth_curvature_bulge_m(d1_m: np.ndarray, d2_m: np.ndarray, k: float) -> np.ndarray:
    Re = 6371000.0
    return (d1_m * d2_m) / (2.0 * k * Re)


def first_fresnel_radius_m(d1_m: np.ndarray, d2_m: np.ndarray, freq_mhz: float) -> np.ndarray:
    lam = wavelength_m(freq_mhz)
    return np.sqrt(lam * d1_m * d2_m / (d1_m + d2_m + 1e-9))


def knife_edge_diffraction_loss_db(nu: np.ndarray) -> np.ndarray:
    return np.where(
        nu <= -0.78,
        0.0,
        6.9 + 20.0 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1.0) + nu - 0.1),
    )


# ---------- Ground presets for Fresnel model ----------
GROUND_DB: dict[str, tuple[float, float]] = {
    "generic": (4.0, 0.02),
    "dry_soil": (3.0, 0.001),
    "wet_soil": (15.0, 0.02),
    "asphalt": (6.0, 0.01),
    "water": (80.0, 4.0),
}


def _complex_eps(eps_r: float, sigma: float, freq_mhz: float) -> complex:
    f = float(freq_mhz) * 1e6
    eps0 = 8.854e-12
    return eps_r - 1j * sigma / (2.0 * np.pi * f * eps0)


def _fresnel_gamma(
    theta_i_rad: np.ndarray, freq_mhz: float, eps_r: float, sigma: float, pol: str
) -> np.ndarray:
    n = np.sqrt(_complex_eps(eps_r, sigma, freq_mhz)).astype(np.complex128)
    sin_ti = np.sin(theta_i_rad).astype(np.complex128)
    cos_ti = np.cos(theta_i_rad).astype(np.complex128)
    sin_tt = sin_ti / n
    sin_tt = np.clip(sin_tt.real, -1.0, 1.0) + 1j * np.clip(sin_tt.imag, -1e6, 1e6)
    cos_tt = np.sqrt(1 - sin_tt**2)
    pol_u = pol.upper()
    if pol_u == "V":
        num = n * cos_ti - cos_tt
        den = n * cos_ti + cos_tt
    else:
        num = cos_ti - n * cos_tt
        den = cos_ti + n * cos_tt
    den = np.where(np.abs(den) < 1e-12, 1e-12 + 0j, den)
    return (num / den).astype(np.complex128)


def two_ray_fresnel_path_loss_db(
    d_m: np.ndarray,
    freq_mhz: float,
    ht_m: float,
    hr_m: float,
    ground_type: str = "generic",
    pol: str = "V",
) -> np.ndarray:
    lam = 3e8 / (float(freq_mhz) * 1e6)
    k = 2.0 * np.pi / lam
    d = np.maximum(d_m.astype(float), 1e-3)
    d_direct = np.sqrt(d**2 + (ht_m - hr_m) ** 2)
    d_refl = np.sqrt(d**2 + (ht_m + hr_m) ** 2)
    delta_r = d_refl - d_direct
    cos_ti = (ht_m + hr_m) / np.sqrt(d**2 + (ht_m + hr_m) ** 2)
    cos_ti = np.clip(cos_ti, 0.0, 1.0)
    theta_i = np.arccos(cos_ti)
    eps_r, sigma = GROUND_DB.get(ground_type, GROUND_DB["generic"])
    Gamma = _fresnel_gamma(theta_i, freq_mhz, eps_r, sigma, pol)
    refl_scale = (d_direct / np.maximum(d_refl, 1e-6)) * np.exp(-1j * k * delta_r)
    field_ratio = 1.0 + Gamma * refl_scale
    mag = np.maximum(np.abs(field_ratio), 1e-6)
    pl_fspl = fspl_db(d_direct, freq_mhz)
    return pl_fspl - 20.0 * np.log10(mag)


# ------------------- Buildings (raycast + rasterize) -------------------
def building_nlos_mask_raycast(  # noqa: C901
    buildings_mask: np.ndarray,
    xmin: float,
    ymin: float,
    ymax: float,
    grid_m: float,
    tx_x: float,
    tx_y: float,
    xx: np.ndarray,
    yy: np.ndarray,
    samples: int = 24,
    tile: int = 256,
    stride: int = 1,
    timeout_s: float | None = None,
) -> np.ndarray:
    t_start = time.perf_counter()

    _safe_twrite("raycast", samples=int(samples), tile=int(tile), stride=int(stride))

    if buildings_mask is None or buildings_mask.size == 0 or not np.any(buildings_mask):
        return np.zeros_like(xx, dtype=bool)

    ny, nx = buildings_mask.shape
    _safe_twrite("raycast", nx=int(nx), ny=int(ny))

    if xx.shape != (ny, nx) or yy.shape != (ny, nx):
        raise ValueError("xx/yy must match buildings_mask shape")

    samples = max(3, int(samples))
    tile = int(max(64, min(1024, tile)))
    stride = int(max(1, stride))
    tile_step = max(1, tile // max(1, stride))
    inv_cell = 1.0 / float(grid_m)

    def world_to_rc(xw: np.ndarray, yw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = (xw - xmin) * inv_cell
        row = (ymax - yw) * inv_cell
        r = np.clip(np.rint(row).astype(np.int32), 0, ny - 1)
        c = np.clip(np.rint(col).astype(np.int32), 0, nx - 1)
        return r, c

    t = np.linspace(0.0, 1.0, samples, dtype=float)[1:-1]
    sparse_h = (ny + stride - 1) // stride
    sparse_w = (nx + stride - 1) // stride
    sparse = np.zeros((sparse_h, sparse_w), dtype=bool)
    logger.info("Raycast: grid=%dx%d samples=%d tile=%d stride=%d", ny, nx, samples, tile, stride)

    yy_coarse = yy[0:ny:stride, 0:nx:stride]
    xx_coarse = xx[0:ny:stride, 0:nx:stride]
    _safe_twrite("raycast", coarse_pixels=int(yy_coarse.size))

    for r0 in range(0, sparse_h, tile_step):
        r1 = min(r0 + tile_step, sparse_h)
        for c0 in range(0, sparse_w, tile_step):
            c1 = min(c0 + tile_step, sparse_w)

            if timeout_s is not None and (time.perf_counter() - t_start) > timeout_s:
                logger.warning(
                    "Raycast aborted by timeout after %.2fs", time.perf_counter() - t_start
                )
                _safe_twrite("raycast", timeout=True)
                out = np.repeat(np.repeat(sparse, stride, axis=0), stride, axis=1)
                return out[:ny, :nx]

            yy_tile = yy_coarse[r0:r1, c0:c1]
            xx_tile = xx_coarse[r0:r1, c0:c1]
            if yy_tile.size == 0 or xx_tile.size == 0:
                continue

            blocked = np.zeros_like(yy_tile, dtype=bool)
            for alpha in t:
                x_line = tx_x + (xx_tile - tx_x) * alpha
                y_line = tx_y + (yy_tile - tx_y) * alpha
                r_idx, c_idx = world_to_rc(x_line, y_line)
                hits = buildings_mask[r_idx, c_idx]
                blocked |= hits
                if blocked.all():
                    break

            sparse[r0:r1, c0:c1] = blocked

        logger.debug(
            "Raycast progress: coarse rows %d/%d (%.1f%%)",
            r1,
            sparse_h,
            100.0 * r1 / max(1, sparse_h),
        )

    out_full = np.repeat(np.repeat(sparse, stride, axis=0), stride, axis=1)
    out_full = out_full[:ny, :nx]
    _safe_twrite(
        "raycast",
        seconds=round(time.perf_counter() - t_start, 3),
        nlos_pixels=int(np.count_nonzero(out_full)),
    )
    logger.info("Raycast done in %.2fs", time.perf_counter() - t_start)
    return out_full


def rasterize_buildings_mask(
    buildings_gdf: gpd.GeoDataFrame,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    grid_m: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    t0 = time.perf_counter()
    _safe_twrite("rasterize", grid=dict(nx=int(nx), ny=int(ny), cell_m=float(grid_m)))

    if buildings_gdf is None or buildings_gdf.empty:
        return np.zeros((ny, nx), dtype=bool)

    bg = buildings_gdf.to_crs(epsg=3857)
    transform = from_origin(xmin, ymax, grid_m, grid_m)
    shapes = [(geom, 1) for geom in bg.geometry if geom is not None and not geom.is_empty]
    _safe_twrite("rasterize", shapes=int(len(shapes)))

    mask = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    _safe_twrite(
        "rasterize", seconds=round(time.perf_counter() - t0, 3), covered_pixels=int(mask.sum())
    )
    return mask.astype(bool)


# ------------------- Terrain LoS/diffraction -------------------
def terrain_los_and_diffraction_db(
    dsm_path: Path | None,
    tx_x: float,
    tx_y: float,
    tx_h_agl_m: float,
    xx: np.ndarray,
    yy: np.ndarray,
    rx_h_agl_m: float,
    freq_mhz: float,
    k: float,
    samples: int = 16,
) -> tuple[np.ndarray | None, np.ndarray]:
    ny, nx = xx.shape
    los_mask = np.ones((ny, nx), dtype=bool)
    Ld = np.zeros((ny, nx), dtype=float)
    if dsm_path is None or not Path(dsm_path).exists():
        return None, Ld

    with rasterio.open(dsm_path) as ds:
        dsm = ds.read(1)
        ds_transform = ds.transform
        ds_crs = ds.crs
        from pyproj import Transformer

        grid_crs = "EPSG:3857"
        if ds_crs and ds_crs.to_string() != grid_crs:
            transformer = Transformer.from_crs(grid_crs, ds_crs, always_xy=True)
            tx_x, tx_y = transformer.transform(tx_x, tx_y)
            xx, yy = transformer.transform(xx, yy)

        def sample_dsm(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
            col = (xp - ds_transform.c) / ds_transform.a
            row = (yp - ds_transform.f) / ds_transform.e
            r0 = np.clip(np.floor(row).astype(int), 0, dsm.shape[0] - 2)
            c0 = np.clip(np.floor(col).astype(int), 0, dsm.shape[1] - 2)
            dr = row - r0
            dc = col - c0
            z00 = dsm[r0, c0]
            z01 = dsm[r0, c0 + 1]
            z10 = dsm[r0 + 1, c0]
            z11 = dsm[r0 + 1, c0 + 1]
            return (
                z00 * (1 - dc) * (1 - dr)
                + z01 * dc * (1 - dr)
                + z10 * (1 - dc) * dr
                + z11 * dc * dr
            )

        t = np.linspace(0.05, 0.95, samples, dtype=float)
        t2 = t.reshape(-1, 1, 1)
        x_line = tx_x + (xx - tx_x) * t2
        y_line = tx_y + (yy - tx_y) * t2
        d_total = np.hypot(xx - tx_x, yy - tx_y)
        d_total = np.maximum(d_total, 1.0)
        d1 = t2 * d_total
        d2 = (1.0 - t2) * d_total
        z_tx = sample_dsm(np.full_like(xx, tx_x), np.full_like(yy, tx_y))
        z_rx = sample_dsm(xx, yy)
        h_tx = z_tx + tx_h_agl_m
        h_rx = z_rx + rx_h_agl_m
        h_line = h_tx + (h_rx - h_tx) * t2
        bulge = earth_curvature_bulge_m(d1, d2, k)
        z_path = sample_dsm(x_line, y_line)
        rF = first_fresnel_radius_m(d1, d2, freq_mhz)
        clearance = (h_line - 0.6 * rF) - (z_path + bulge)
        worst = np.max(-(clearance), axis=0)
        d1m = d_total * 0.5
        d2m = d_total * 0.5
        lam = wavelength_m(freq_mhz)
        nu = worst * np.sqrt(2.0 / lam * (d1m + d2m) / (d1m * d2m + 1e-9))
        nu = np.maximum(nu, -5.0)
        Ld = knife_edge_diffraction_loss_db(nu)
        los_mask = worst <= 0.0
        return los_mask, Ld


# ------------------- Path loss grid (with optional terrain/buildings) -------------------
def compute_path_loss_grid(
    model: str,
    d_m: np.ndarray,
    freq_mhz: float,
    ht_m: float,
    hr_m: float,
    tx_x: float,
    tx_y: float,
    xx: np.ndarray,
    yy: np.ndarray,
    k: float,
    dsm_path: Path | None,
    buildings_mask: np.ndarray | None,
    building_loss_db: float,
    ground_type: str = "generic",
    polarization: str = "V",
) -> tuple[np.ndarray, np.ndarray | None]:
    if model != "Terrain (LoS + diffraction)":
        st.session_state["_last_los_mask"] = None

    if model == "FSPL (Free Space)":
        pl = fspl_db(d_m, freq_mhz)
        return pl, None

    if model == "Two-Ray (Interference)":
        pl = two_ray_path_loss_db(d_m, freq_mhz, ht_m, hr_m)
        return pl, None

    if model == "Two-Ray (Fresnel ground)":
        pl = two_ray_fresnel_path_loss_db(d_m, freq_mhz, ht_m, hr_m, ground_type, polarization)
        return pl, None

    pl_fspl = fspl_db(d_m, freq_mhz)
    if dsm_path is None or not Path(dsm_path).exists():
        los_mask = None
        Ld = np.zeros_like(pl_fspl)
    else:
        los_mask, Ld = terrain_los_and_diffraction_db(
            dsm_path=dsm_path,
            tx_x=tx_x,
            tx_y=tx_y,
            tx_h_agl_m=ht_m,
            xx=xx,
            yy=yy,
            rx_h_agl_m=hr_m,
            freq_mhz=freq_mhz,
            k=k,
            samples=16,
        )
    st.session_state["_last_los_mask"] = los_mask
    return pl_fspl + Ld, los_mask


def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _project_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        return oxproj.project_gdf(gdf)
    except Exception:
        try:
            return gdf.to_crs(gdf.estimate_utm_crs())
        except Exception:
            return gdf


def download_dem_opentopo_cop30(
    bbox: tuple[float, float, float, float], out_path: Path, api_key: str
) -> bool:
    api_key = (api_key or "").strip().strip("<>").strip('"').strip("'")
    if not api_key:
        st.error("OpenTopography API key missing.")
        return False
    lat_s, lat_n, lon_w, lon_e = bbox
    url = "https://portal.opentopography.org/API/globaldem"
    params: dict[str, str] = {
        "demtype": "COP30",
        "south": f"{lat_s:.6f}",
        "north": f"{lat_n:.6f}",
        "west": f"{lon_w:.6f}",
        "east": f"{lon_e:.6f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    st.info("Downloading DEM from OpenTopography (COP30)...")
    r = requests.get(url, params=params, stream=True, timeout=300)
    try:
        r.raise_for_status()
    except Exception as e:
        st.error(f"Download failed: {e}\nResponse: {r.text[:400]}")
        return False
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    _safe_twrite("io", dsm_saved=str(out_path))
    return True


# ------------------- AOI & DSM -------------------
city_center, bbox = geocode_city_with_buffer_km(city, float(aoi_radius_km))
logger.info(
    "[SESSION %s] AOI city=%s radius_km=%.2f center=(%.6f, %.6f) bbox(lat_s=%.6f, lat_n=%.6f, lon_w=%.6f, lon_e=%.6f)",
    SESSION_ID,
    city,
    float(aoi_radius_km),
    city_center[0],
    city_center[1],
    bbox[0],
    bbox[1],
    bbox[2],
    bbox[3],
)
dsm_path = bbox_to_dsm_path(bbox)

# ------------------- OSM local-first helpers -------------------


# ------------------- OSM bbox helpers -------------------
def _shrink_bbox(
    bbox: tuple[float, float, float, float], ratio: float
) -> tuple[float, float, float, float]:
    """Shrink bbox towards its center by *ratio* (0<ratio<=1). Keeps center fixed."""
    lat_s, lat_n, lon_w, lon_e = bbox
    lat_c = (lat_s + lat_n) / 2.0
    lon_c = (lon_w + lon_e) / 2.0
    dlat = (lat_n - lat_s) / 2.0
    dlon = (lon_e - lon_w) / 2.0
    ratio = max(0.05, min(1.0, float(ratio)))
    return (lat_c - dlat * ratio, lat_c + dlat * ratio, lon_c - dlon * ratio, lon_c + dlon * ratio)


_OVERPASS_POOL: list[str] = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


def _osmnx_set_timeout(timeout_s: int) -> None:
    try:
        ox.settings.use_cache = True
        ox.settings.overpass_settings = f"[out:json][timeout:{int(timeout_s)}]"
        ox.settings.requests_timeout = int(timeout_s)
        request_kwargs = dict(getattr(ox.settings, "requests_kwargs", {}) or {})
        request_kwargs.pop("timeout", None)
        ox.settings.requests_kwargs = request_kwargs
        ox.settings.overpass_rate_limit = True
    except Exception:
        logger.debug("Could not set OSMnx overpass settings", exc_info=True)


def _osmnx_fetch_buildings_core(
    north: float, south: float, east: float, west: float, tags: dict
) -> gpd.GeoDataFrame:
    if hasattr(ox, "geometries_from_bbox"):
        return ox.geometries_from_bbox(north, south, east, west, tags)

    # OSMnx 2.x expects bbox=(left, bottom, right, top).
    feat_fn: Callable[..., gpd.GeoDataFrame] = cast(
        Callable[..., gpd.GeoDataFrame], ox.features_from_bbox
    )
    return feat_fn((west, south, east, north), tags)


def _osmnx_fetch_buildings_safe(
    north: float,
    south: float,
    east: float,
    west: float,
    tags: dict,
    timeout_s: int,
    endpoints: list[str] | None = None,
) -> tuple[gpd.GeoDataFrame, str | None]:
    """
    Rotate across Overpass mirrors with per-attempt timeouts.
    Runs the core OSMnx fetch in a worker thread to avoid blocking the UI.
    Returns (GeoDataFrame, working_endpoint | None). If all fail, returns (empty_gdf, "OverpassFetchFailed").
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    from requests.exceptions import ConnectionError as ReqConnError
    from requests.exceptions import Timeout

    # Inline default Overpass mirror pool (used if `endpoints` is None)
    DEFAULT_OVERPASS_POOL: list[str] = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
    ]

    eps = list(endpoints or DEFAULT_OVERPASS_POOL)
    if not eps:
        eps = [getattr(ox.settings, "overpass_endpoint", "https://overpass-api.de/api/interpreter")]

    # Ensure sane timeouts
    t_client = int(max(3, timeout_s))

    for i, ep in enumerate(eps, 1):
        try:
            # Configure endpoint + server/client-side timeouts for this attempt
            try:
                # OSMnx 2.x expects a base Overpass URL and appends
                # "/interpreter" itself.
                base_url = ep.rstrip("/")
                if base_url.endswith("/interpreter"):
                    base_url = base_url[: -len("/interpreter")]

                ox.settings.overpass_url = base_url
                ox.settings.overpass_settings = f"[out:json][timeout:{t_client}]"
                ox.settings.requests_timeout = t_client

                request_kwargs = dict(
                    getattr(ox.settings, "requests_kwargs", {}) or {}
                )
                request_kwargs.pop("timeout", None)
                ox.settings.requests_kwargs = request_kwargs
                ox.settings.overpass_rate_limit = True
            except Exception:
                # Non-fatal; continue with defaults
                pass

            _safe_twrite("osm", attempt=i, endpoint=ep, client_timeout_s=t_client)

            # Run the blocking OSMnx call in a worker with a client timeout.
            # Do not use the executor as a context manager here: its __exit__
            # waits for a running worker even after fut.result() times out.
            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(
                _osmnx_fetch_buildings_core,
                north,
                south,
                east,
                west,
                tags,
            )
            try:
                g = fut.result(timeout=t_client)
            except FuturesTimeout:
                fut.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception:
                ex.shutdown(wait=True)
                raise
            else:
                ex.shutdown(wait=True)

            if g is None or getattr(g, "empty", True):
                # Treat None/empty as non-fatal; try next endpoint
                _safe_twrite("osm", attempt=i, endpoint=ep, empty=True)
                logger.warning("Overpass returned empty on %s (attempt %d)", ep, i)
                continue

            # Success
            _safe_twrite("osm", attempt=i, endpoint=ep, ok=True, rows=int(len(g)))
            return g, ep

        except FuturesTimeout:
            _safe_twrite("osm", attempt=i, endpoint=ep, timeout=True)
            logger.warning("Overpass client timeout on %s (attempt %d)", ep, i)
            continue
        except (Timeout, ReqConnError) as e:
            _safe_twrite("osm", attempt=i, endpoint=ep, error=type(e).__name__)
            logger.warning("Overpass error %s on %s (attempt %d)", type(e).__name__, ep, i)
            continue
        except Exception as e:
            _safe_twrite("osm", attempt=i, endpoint=ep, error=type(e).__name__)
            logger.exception("Overpass unexpected error on %s (attempt %d)", ep, i)
            continue

    # All attempts failed
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), "OverpassFetchFailed"


def _clean_buildings_4326(
    g: gpd.GeoDataFrame,
    min_area_m2: float = 2.0,
    simplify_tol_m: float = 0.75,
    max_feats: int = 20000,
) -> gpd.GeoDataFrame:
    b = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if b.empty:
        return gpd.GeoDataFrame(geometry=[], crs=(getattr(b, "crs", None) or "EPSG:4326"))

    if b.crs is None:
        b = b.set_crs("EPSG:4326")

    b3857 = b.to_crs("EPSG:3857")
    b3857["geometry"] = b3857["geometry"].buffer(0)
    b3857 = b3857[b3857.is_valid]
    b3857 = b3857[~b3857.is_empty]
    b3857 = b3857[b3857.area >= float(min_area_m2)]
    b3857["geometry"] = b3857["geometry"].simplify(float(simplify_tol_m), preserve_topology=True)

    if len(b3857) > int(max_feats):
        b3857 = (
            b3857.assign(_area=b3857.area)
            .sort_values("_area", ascending=False)
            .head(int(max_feats))
            .drop(columns=["_area"])
        )

    return b3857.to_crs("EPSG:4326")[["geometry"]].dropna()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _osm_local_path(
    city_label: str, bbox: tuple[float, float, float, float], radius_km: float
) -> Path:
    lat_s, lat_n, lon_w, lon_e = bbox
    fname = (
        f"osm_buildings_{_slug(city_label)}_r{radius_km:.2f}km_"
        f"{lat_s:.5f}_{lat_n:.5f}_{lon_w:.5f}_{lon_e:.5f}.gpkg"
    )
    return osm_dir / fname


def _save_local_buildings(b: gpd.GeoDataFrame, fp: Path) -> Path:
    try:
        b.to_file(fp, driver="GPKG", layer="buildings")
        return fp
    except Exception:
        alt = fp.with_suffix(".geojson")
        b.to_file(alt, driver="GeoJSON")
        return alt


def _load_local_buildings(fp: Path) -> gpd.GeoDataFrame:
    if not fp.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    g = gpd.read_file(fp)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    else:
        g = g.to_crs("EPSG:4326")
    return g[["geometry"]].dropna()


# ------------------- OSM buildings fetch -------------------
def fetch_osm_buildings_bbox(
    lat_s: float, lat_n: float, lon_w: float, lon_e: float
) -> gpd.GeoDataFrame:
    osm_timeout_s: int = int(st.session_state.get("osm_timeout_s", 30))
    skip_if_slow: bool = bool(st.session_state.get("skip_if_slow", True))

    north, south, east, west = float(lat_n), float(lat_s), float(lon_e), float(lon_w)
    tags: dict[str, bool | str | list[str]] = {"building": True}

    _safe_twrite(
        "osm",
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        bbox=dict(n=north, s=south, e=east, w=west),
        timeout_s=int(osm_timeout_s),
    )

    _osmnx_set_timeout(osm_timeout_s)
    logger.info(
        "OSM fetch: bbox(n=%.6f s=%.6f e=%.6f w=%.6f) timeout=%ss",
        north,
        south,
        east,
        west,
        osm_timeout_s,
    )

    t0 = time.perf_counter()
    t_fetch = time.perf_counter()
    g, err = _osmnx_fetch_buildings_safe(
        north, south, east, west, tags, timeout_s=osm_timeout_s, endpoints=_OVERPASS_POOL
    )
    fetch_dt = round(time.perf_counter() - t_fetch, 3)
    raw_rows = 0 if g is None else int(len(g))
    logger.info("OSM fetch completed in %.3fs; raw rows=%d", fetch_dt, raw_rows)
    _safe_twrite("osm", fetch_seconds=fetch_dt, raw_rows=raw_rows)

    if err is not None:
        _safe_twrite("osm", error=err)
        if skip_if_slow:
            st.warning(
                "Overpass is slow or unreachable. Skipping building attenuation for this run."
            )
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if g is None or g.empty:
        _safe_twrite(
            "osm",
            polygons_before=0,
            polygons_after=0,
            post_seconds=round(time.perf_counter() - t0, 3),
        )
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    polygons_before = int(len(g[g.geom_type.isin(["Polygon", "MultiPolygon"])]))
    bout = _clean_buildings_4326(g)

    post_dt = round(time.perf_counter() - t0, 3)
    logger.info("OSM post-process: polygons=%d in %.3fs", len(bout), post_dt)
    _safe_twrite(
        "osm",
        polygons_before=polygons_before,
        polygons_after=int(len(bout)),
        post_seconds=post_dt,
    )
    return bout


with st.sidebar.expander("OSM Buildings Controls", expanded=False):
    st.slider(
        "Max AOI radius for buildings (km)",
        min_value=0.5,
        max_value=6.0,
        step=0.5,
        value=float(st.session_state.get("max_aoi_km_for_buildings", 3.0)),
        key="max_aoi_km_for_buildings",
        help="Apply buildings only when AOI radius ≤ this value.",
    )

    st.checkbox(
        "Skip OSM if timeout/error",
        key="skip_if_slow",
        value=bool(st.session_state.get("skip_if_slow", True)),
    )

    st.slider(
        "UI wait cap for OSM (s)",
        min_value=5,
        max_value=60,
        step=5,
        value=int(st.session_state.get("osm_ui_wait_cap_s", 20)),
        key="osm_ui_wait_cap_s",
        help="Max seconds the UI will wait for an Overpass response before skipping this run.",
    )

    st.checkbox(
        "Auto-shrink AOI on OSM failure",
        key="auto_shrink_osm",
        value=bool(st.session_state.get("auto_shrink_osm", True)),
        help="Retry Overpass with smaller bbox if the initial fetch returns empty or times out.",
    )

# ------------------- Local-first orchestration -------------------
with st.sidebar.expander("OSM Buildings Source", expanded=False):
    osm_source_mode = st.radio(
        "Source",
        ["Local only", "Local first, then Overpass", "Overpass only"],
        index=1,
        help="Where to load building footprints from.",
    )
    osm_save_after_fetch = st.checkbox(
        "Save fetched buildings to data folder",
        value=True,
        help="Store cleaned buildings under ~/manet-heatmap/data for reuse.",
    )


def _fetch_osm_bbox_with_ui_cap(
    bbox: tuple[float, float, float, float], ui_wait_s: float
) -> gpd.GeoDataFrame:
    """Run fetch_osm_buildings_bbox in a worker thread; return empty if the UI cap elapses."""
    lat_s, lat_n, lon_w, lon_e = bbox
    res_holder: dict[str, gpd.GeoDataFrame] = {}

    def _runner():
        try:
            res_holder["gdf"] = fetch_osm_buildings_bbox(lat_s, lat_n, lon_w, lon_e)
        except Exception:
            res_holder["gdf"] = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_runner)
    try:
        fut.result(timeout=float(max(1.0, ui_wait_s)))
    except FuturesTimeout:
        _safe_twrite("osm", ui_wait_timeout=True, cap_s=float(ui_wait_s))
        fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        ex.shutdown(wait=True)

    return res_holder.get("gdf", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))


def _fetch_buildings_with_shrink(
    bbox: tuple[float, float, float, float],
    radius_km: float,
    auto_shrink: bool,
    max_aoi_km: float,
    source_mode: str,
    save_after_fetch: bool,
    city_label: str,
) -> gpd.GeoDataFrame:
    """Attempt local/remote fetch; optionally retry Overpass with smaller bbox ratios."""
    # First try through the existing orchestration (Local/Remote) with the original bbox.
    g0 = get_osm_buildings_local_or_remote(
        city_label=city_label,
        bbox=bbox,
        radius_km=radius_km,
        source_mode=source_mode,
        save_after_fetch=save_after_fetch,
    )
    if g0 is not None and not g0.empty:
        return g0

    # If source is Local only or auto-shrink disabled, stop.
    if (source_mode == "Local only") or (not auto_shrink):
        return g0 if g0 is not None else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Shrink ladder (try smaller AOIs; fetch via Overpass directly to avoid mismatched local cache)
    ladder = [0.9, 0.75, 0.6, 0.5]
    for r in ladder:
        if (radius_km * r) > (max_aoi_km + 1e-6):
            continue
        inner_bbox = _shrink_bbox(bbox, r)
        _safe_twrite("osm", shrink_ratio=float(r))
        # Direct remote fetch, bounded by the configured UI wait cap.
        ui_wait_s = float(st.session_state.get("osm_ui_wait_cap_s", 20))
        g_remote = _fetch_osm_bbox_with_ui_cap(inner_bbox, ui_wait_s)
        if g_remote is not None and not g_remote.empty:
            if save_after_fetch:
                # Save under the inner bbox path for reuse
                fp = _osm_local_path(city_label, inner_bbox, radius_km * r)
                out_fp = _save_local_buildings(g_remote, fp)
                _safe_twrite("osm", saved_path=str(out_fp), shrink_saved=True)
            # Inform the user we shrunk
            st.caption(
                f"OSM buildings fetched with smaller AOI ratio {r:.2f} due to Overpass limits."
            )
            return g_remote

    # Nothing found
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def get_osm_buildings_local_or_remote(
    city_label: str,
    bbox: tuple[float, float, float, float],
    radius_km: float,
    source_mode: str,
    save_after_fetch: bool,
) -> gpd.GeoDataFrame:
    fp = _osm_local_path(city_label, bbox, radius_km)

    if source_mode in ("Local only", "Local first, then Overpass"):
        if fp.exists():
            g_local = _load_local_buildings(fp)
            _safe_twrite("osm", source="local", path=str(fp), count=int(len(g_local)))
            logger.info("Loaded local OSM buildings: %s (count=%d)", fp.name, int(len(g_local)))
            return g_local
        if source_mode == "Local only":
            st.warning("No local OSM buildings file found; continuing without buildings.")
            _safe_twrite("osm", source="local", path=str(fp), missing=True)
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Remote fetch
    ui_wait_s = float(st.session_state.get("osm_ui_wait_cap_s", 20))
    b_remote = _fetch_osm_bbox_with_ui_cap(bbox, ui_wait_s)
    _safe_twrite("osm", source="overpass", fetched=int(len(b_remote)))

    if save_after_fetch and not b_remote.empty:
        out_fp = _save_local_buildings(b_remote, fp)
        _safe_twrite("osm", saved_path=str(out_fp))

    return b_remote


# ------------------- Fetch buildings if requested -------------------
max_aoi_km_for_buildings = float(st.session_state.get("max_aoi_km_for_buildings", 3.0))
buildings_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[])

bbox_sig = tuple(float(round(v, 6)) for v in bbox)
prev_sig = st.session_state.get("_bldg_bbox_sig")
prev_gdf = st.session_state.get("_bldg_gdf")

if apply_bldg and aoi_radius_km <= max_aoi_km_for_buildings:
    # Only reuse non-empty cached data
    if prev_sig == bbox_sig and isinstance(prev_gdf, gpd.GeoDataFrame) and not prev_gdf.empty:
        buildings_gdf = prev_gdf
        logger.info(
            "[SESSION %s] Reusing buildings from session cache; count=%d",
            SESSION_ID,
            int(len(buildings_gdf)),
        )
        _safe_twrite("osm", cached=True, cached_count=int(len(buildings_gdf)))
    else:
        with st.spinner("Loading OSM buildings (local/Overpass)…"):
            city_label = str(st.session_state.get("city_name", city)) or "city"
            buildings_gdf = _fetch_buildings_with_shrink(
                bbox=bbox,
                radius_km=float(aoi_radius_km),
                auto_shrink=bool(st.session_state.get("auto_shrink_osm", True)),
                max_aoi_km=float(max_aoi_km_for_buildings),
                source_mode=osm_source_mode,
                save_after_fetch=osm_save_after_fetch,
                city_label=city_label,
            )
        if buildings_gdf.crs is None:
            buildings_gdf = buildings_gdf.set_crs("EPSG:4326")
        if not buildings_gdf.empty:
            st.session_state["_bldg_bbox_sig"] = bbox_sig
            st.session_state["_bldg_gdf"] = buildings_gdf
        logger.info(
            "[SESSION %s] Buildings ready count=%d crs=%s",
            SESSION_ID,
            int(len(buildings_gdf)),
            getattr(buildings_gdf, "crs", None),
        )
        if buildings_gdf.empty:
            st.info(
                "No buildings found for this AOI or Overpass timed out. Try smaller AOI or higher timeout."
            )
else:
    if apply_bldg and aoi_radius_km > max_aoi_km_for_buildings:
        st.warning(
            "AOI too large for building attenuation; reduce radius or adjust the slider in OSM Buildings Controls."
        )
        _safe_twrite("osm", skipped_too_large=True, radius_km=float(aoi_radius_km))
    buildings_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

# ------------------- DSM Button -------------------

st.sidebar.header("Download DSM Data")
if st.sidebar.button("Fetch DSM (COP30) for city AOI"):
    st.info("Fetching DSM data for the current AOI bbox...")
    api_key = get_api_key()
    if api_key:
        try:
            if download_dem_opentopo_cop30(bbox, dsm_path, api_key):
                st.success(f"DSM saved to {dsm_path}")
            else:
                st.error("Error downloading the DSM data.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("API key is required to fetch DSM data.")

# ------------------- Map Display -------------------
st.sidebar.header("Map Display")
st.sidebar.toggle("Show LoS mask (terrain mode)", key="show_los_mask", value=False)
st.sidebar.toggle("Show Map", key="show_map", value=True)

show_preview = st.session_state.get("show_map", True) and not st.session_state.get("res")
base_map: folium.Map | None = None

if show_preview:
    st.subheader(f"Map for {city}")
    lat_lon: tuple[float, float] | None = (
        (float(city_center[0]), float(city_center[1])) if city_center else None
    )
    if lat_lon is None:
        st.info("Enter a city to display the map.")
    else:
        lat, lon = lat_lon
        base_map = folium.Map(
            location=[lat, lon], zoom_start=12, tiles="OpenStreetMap", control_scale=True
        )
        folium.Marker(
            [lat, lon],
            tooltip=f"TX: {lat:.6f}, {lon:.6f}",
            icon=folium.Icon(color="blue", icon="wifi", prefix="fa"),
        ).add_to(base_map)
        folium.Circle(
            [lat, lon], radius=float(aoi_radius_km) * 1000.0, color="#3388ff", weight=1, fill=False
        ).add_to(base_map)
        st_folium(base_map, height=600, use_container_width=True, key="base_map")


# ------------------- Core simulation -------------------
def quick_simulate(  # noqa: C901
    city_name: str,
    aoi_km: float,
    grid_m: float | int,
    freq_mhz: float,
    bw_mhz: float,
    tx_power_dbm: float,
    tx_gain_db: float,
    rx_gain_db: float,
    rx_nf_db: float,
    target_mbps: float,
    out_root: Path | None = None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    model_choice: str = "FSPL (Free Space)",
    tx_height_m: float = 1.5,
    rx_height_m: float = 1.5,
    k_factor: float = 1.33,
    dsm_path: Path | None = None,
    building_loss_db: float = 0.0,
    ground_type: str = "generic",
    polarization: str = "V",
    enable_bldg_raycast: bool = False,
    bldg_nlos_penalty_db: float = 25.0,
    return_grid: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    tx_lat, tx_lon = ox.geocode(city_name)
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    invproj = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    tx_x, tx_y = proj.transform(tx_lon, tx_lat)

    radius_m = float(aoi_km) * 1000.0
    xmin, xmax = tx_x - radius_m, tx_x + radius_m
    ymin, ymax = tx_y - radius_m, tx_y + radius_m

    cell_m: float = float(grid_m)
    nx: int = int(np.ceil((xmax - xmin) / cell_m))
    ny: int = int(np.ceil((ymax - ymin) / cell_m))

    max_cells = 1200 * 1200
    if nx * ny > max_cells:
        step_up: float = float(np.ceil(np.sqrt((nx * ny) / max_cells)))
        cell_m *= step_up
        nx = int(np.ceil((xmax - xmin) / cell_m))
        ny = int(np.ceil((ymax - ymin) / cell_m))

    xs = xmin + np.arange(nx) * cell_m + cell_m / 2
    ys = ymax - np.arange(ny) * cell_m - cell_m / 2
    xx, yy = np.meshgrid(xs, ys)
    d_m = np.hypot(xx - tx_x, yy - tx_y)

    # Buildings mask
    if (
        (buildings_gdf is not None)
        and (not buildings_gdf.empty)
        and st.session_state.get("apply_bldg_loss", False)
    ):
        with time_block("buildings_rasterize"):
            buildings_mask = rasterize_buildings_mask(
                buildings_gdf, xmin, ymin, xmax, ymax, cell_m, nx, ny
            )
        log_resources("after_rasterize")
    else:
        buildings_mask = None

    dsm_effective: Path | None = dsm_path if (dsm_path is not None and dsm_path.exists()) else None

    pl_db, los_mask = compute_path_loss_grid(
        model=model_choice,
        d_m=d_m,
        freq_mhz=freq_mhz,
        ht_m=tx_height_m,
        hr_m=rx_height_m,
        tx_x=tx_x,
        tx_y=tx_y,
        xx=xx,
        yy=yy,
        k=k_factor,
        dsm_path=dsm_effective,
        buildings_mask=buildings_mask,
        building_loss_db=building_loss_db,
        ground_type=ground_type,
        polarization=polarization,
    )

    sanity_check_physics("single" if not return_grid else "grid", float(freq_mhz), d_m, pl_db)

    if (buildings_mask is not None) and st.session_state.get("apply_bldg_loss", False):
        pl_db = pl_db + (buildings_mask.astype(float) * float(building_loss_db))

    if enable_bldg_raycast and (buildings_mask is not None):
        nlos_mask = building_nlos_mask_raycast(
            buildings_mask=buildings_mask,
            xmin=xmin,
            ymin=ymin,
            ymax=ymax,
            grid_m=cell_m,
            tx_x=tx_x,
            tx_y=tx_y,
            xx=xx,
            yy=yy,
            samples=int(st.session_state.get("bldg_samples", 24)),
            tile=int(st.session_state.get("bldg_tile", 256)),
            stride=int(st.session_state.get("bldg_stride", 1)),
            timeout_s=float(
                min(
                    float(st.session_state.get("raycast_timeout_s", 8.0)),
                    float(st.session_state.get("raycast_time_budget_s", 8.0)),
                )
            ),
        )
        st.session_state["_last_los_mask"] = nlos_mask
        pl_db = pl_db + (nlos_mask.astype(float) * float(bldg_nlos_penalty_db))

    rx_p_dbm = (tx_power_dbm + tx_gain_db + rx_gain_db) - pl_db
    n_dbm = noise_floor_dbm(bw_mhz, rx_nf_db)
    sinr_db = rx_p_dbm - n_dbm
    se = spectral_efficiency_from_sinr_db(sinr_db)
    thr_mbps = se * bw_mhz
    thr_mbps = np.where(d_m <= radius_m, thr_mbps, np.nan)

    finite = np.isfinite(thr_mbps)
    if np.any(finite):
        logger.info(
            "Throughput stats: min=%.3f median=%.3f max=%.3f",
            float(np.nanmin(thr_mbps)),
            float(np.nanmedian(thr_mbps)),
            float(np.nanmax(thr_mbps)),
        )

    meets = thr_mbps >= target_mbps
    max_dist_m = float(np.nanmax(np.where(meets, d_m, np.nan))) if np.any(meets) else 0.0
    max_dist_km = max_dist_m / 1000.0

    out_dir = Path(out_root or (Path.home() / "manet-heatmap" / "data" / "outputs"))
    ensure_dir(out_dir)
    transform = from_origin(xmin, ymax, cell_m, cell_m)
    geotiff_path = str(out_dir / "throughput_quick.tif")
    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=thr_mbps.shape[0],
        width=thr_mbps.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        compress="lzw",
        nodata=np.nan,
    ) as dst:
        dst.write(thr_mbps.astype("float32"), 1)
    _safe_twrite("io", geotiff=str(geotiff_path))

    center_row = thr_mbps.shape[0] // 2
    invproj2 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lonlats = [invproj2.transform(x, ys[center_row]) for x in xs]
    csv_path = str(out_dir / "throughput_centerline.csv")
    pd.DataFrame(
        {
            "lat": [lat for lon, lat in lonlats],
            "lon": [lon for lon, lat in lonlats],
            "throughput_Mbps": thr_mbps[center_row, :],
        }
    ).to_csv(csv_path, index=False)
    _safe_twrite("io", csv=str(csv_path))

    log_resources("after_outputs")

    w_lon_min, w_lat_max = invproj.transform(xmin, ymax)
    w_lon_max, w_lat_min = invproj.transform(xmax, ymin)
    bounds: list[list[float]] = [
        [float(w_lat_min), float(w_lon_min)],
        [float(w_lat_max), float(w_lon_max)],
    ]

    h, w = thr_mbps.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if np.any(finite):
        vmin = float(np.nanpercentile(thr_mbps[finite], 2))
        vmax = float(np.nanpercentile(thr_mbps[finite], 98))
        if vmin >= vmax:
            vmax = vmin + 1e-6
        scaled = np.zeros_like(thr_mbps, dtype=float)
        scaled[finite] = np.clip((thr_mbps[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
        img[..., 0] = (scaled * 255).astype(np.uint8)
        img[..., 1] = np.minimum(255, (scaled * 255 + 100)).astype(np.uint8)
        img[..., 2] = (255 - (scaled * 255)).astype(np.uint8)
        img[..., 3] = np.where(finite, 200, 0).astype(np.uint8)
    else:
        img[..., 0] = 255
    img[..., 3] = np.where(finite, 200, 120).astype(np.uint8)

    tx_lon2, tx_lat2 = invproj.transform(tx_x, tx_y)

    res: dict[str, Any] = {
        "image": img,
        "bounds": bounds,
        "tx_lat": float(tx_lat2),
        "tx_lon": float(tx_lon2),
        "geotiff": geotiff_path,
        "csv": csv_path,
        "max_dist_km": float(max_dist_km),
    }
    if return_grid:
        res["thr_grid"] = thr_mbps
        res["d_grid_m"] = d_m
    return res


# ------------------- Signature-based state clear -------------------
_sig = (
    bool(combine_worst),
    tuple(combine_models),
    str(model_choice),
    float(aoi_radius_km),
    float(grid_resolution_m),
    int(frequency_mhz),
    float(tx_height_m),
    float(rx_height_m),
    str(ground_type),
    str(polarization),
)
if st.session_state.get("_last_sig") != _sig:
    logger.info(
        "Mode signature changed. Clearing cached result and LoS mask. prev=%r now=%r",
        st.session_state.get("_last_sig"),
        _sig,
    )
    st.session_state["_last_sig"] = _sig
    st.session_state["res"] = None
    st.session_state["_last_los_mask"] = None


# ------------------- Run Simulation -------------------
@contextmanager
def run_watchdog(seconds: float, label: str) -> Iterator[None]:
    def _trip() -> None:
        try:
            st.session_state["_sim_running"] = False
            st.session_state["_watchdog_tle"] = True
        except Exception:
            logger.exception("Watchdog unlock failed")
        logger.error("[WATCHDOG] %s exceeded %.1fs; UI unlocked.", label, seconds)

    timer = None
    try:
        if seconds and seconds > 0:
            from threading import Timer as _Timer

            timer = _Timer(seconds, _trip)
            timer.daemon = True
            timer.start()
        yield
    finally:
        if timer is not None:
            timer.cancel()


st.sidebar.header("Run Simulation")
if st.session_state.get("_sim_running", False):
    if st.sidebar.button(
        "Force unlock", key="force_unlock_btn", help="Use if a previous run stalled"
    ):
        st.session_state["_sim_running"] = False
        st.session_state["run_sim"] = False
        st.toast("Unlocked.")

_run_disabled = bool(st.session_state.get("_sim_running", False))
if st.sidebar.button("Run Simulation", key="run_sim_btn", disabled=_run_disabled):
    st.session_state["run_sim"] = True

_watchdog_s: float = float(st.session_state.get("watchdog_seconds", 240.0))
st.session_state.setdefault("_watchdog_tle", False)

if st.session_state.get("run_sim") and not st.session_state.get("_sim_running"):
    st.session_state["_sim_running"] = True
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(
        "[RUN %s] start combine_worst=%s models=%s model_choice=%s",
        run_id,
        combine_worst,
        combine_models,
        model_choice,
    )
    try:
        # DSM info
        if dsm_path.exists():
            dsm, dsm_crs, dsm_bounds = cached_load_dsm(dsm_path)
            logger.info(
                "[RUN %s] DSM loaded path=%s crs=%s bounds=%s",
                run_id,
                dsm_path,
                dsm_crs,
                dsm_bounds,
            )
            arr = dsm[np.isfinite(dsm)]
            if arr.size:
                logger.info(
                    "[RUN %s] DSM elevation range (m): %.2f..%.2f",
                    run_id,
                    float(arr.min()),
                    float(arr.max()),
                )
        else:
            logger.info("[RUN %s] DSM not found -> terrain penalties inactive", run_id)

        params = dict(
            city=city,
            aoi_km=aoi_radius_km,
            grid_m=grid_resolution_m,
            freq_mhz=frequency_mhz,
            bw_mhz=bandwidth_mhz,
            tx_power_dbm=tx_power_dbm,
            tx_gain_db=tx_gain_dbi,
            rx_gain_db=rx_gain_dbi,
            rx_nf_db=7.0,
            target_mbps=target_mbps,
            apply_bldg=apply_bldg,
            building_loss_db=building_loss_db,
            enable_bldg_raycast=enable_bldg_raycast,
            bldg_nlos_penalty_db=bldg_nlos_penalty_db,
        )
        log_run_context(f"RUN {run_id}", params)
        log_resources("pre_run")

        with run_watchdog(_watchdog_s, f"simulation {run_id}"):
            with st.status(f"Running simulation… (run_id={run_id})", expanded=False) as status:
                t0 = time.time()

                if not combine_worst:
                    status.update(label=f"Running: {model_choice}")
                    res = quick_simulate(
                        city_name=city,
                        aoi_km=aoi_radius_km,
                        grid_m=grid_resolution_m,
                        freq_mhz=frequency_mhz,
                        bw_mhz=bandwidth_mhz,
                        tx_power_dbm=tx_power_dbm,
                        tx_gain_db=tx_gain_dbi,
                        rx_gain_db=rx_gain_dbi,
                        rx_nf_db=7.0,
                        target_mbps=target_mbps,
                        out_root=data_root / "outputs",
                        buildings_gdf=buildings_gdf,
                        model_choice=model_choice,
                        tx_height_m=tx_height_m,
                        rx_height_m=rx_height_m,
                        k_factor=k_factor,
                        dsm_path=dsm_path,
                        building_loss_db=building_loss_db,
                        ground_type=ground_type,
                        polarization=polarization,
                        enable_bldg_raycast=enable_bldg_raycast,
                        bldg_nlos_penalty_db=bldg_nlos_penalty_db,
                        return_grid=False,
                        run_id=run_id,
                    )
                else:
                    res_list: list[dict[str, Any]] = []
                    thr_list: list[np.ndarray] = []
                    first: dict[str, Any] | None = None

                    total = len(combine_models)
                    for i, m in enumerate(combine_models, start=1):
                        status.update(label=f"Running model {i}/{total}: {m}")
                        with time_block(f"simulate_{m}"):
                            r = quick_simulate(
                                city_name=city,
                                aoi_km=aoi_radius_km,
                                grid_m=grid_resolution_m,
                                freq_mhz=frequency_mhz,
                                bw_mhz=bandwidth_mhz,
                                tx_power_dbm=tx_power_dbm,
                                tx_gain_db=tx_gain_dbi,
                                rx_gain_db=rx_gain_dbi,
                                rx_nf_db=7.0,
                                target_mbps=target_mbps,
                                out_root=data_root / "outputs",
                                buildings_gdf=buildings_gdf,
                                model_choice=m,
                                tx_height_m=tx_height_m,
                                rx_height_m=rx_height_m,
                                k_factor=k_factor,
                                dsm_path=dsm_path,
                                building_loss_db=building_loss_db,
                                ground_type=ground_type,
                                polarization=polarization,
                                enable_bldg_raycast=enable_bldg_raycast,
                                bldg_nlos_penalty_db=bldg_nlos_penalty_db,
                                return_grid=True,
                                run_id=f"{run_id}-{m.replace(' ', '_')}",
                            )
                        res_list.append(r)
                        thr_list.append(r["thr_grid"])
                        if first is None:
                            first = r
                        log_resources(f"after_{i}_{m}")

                    if not thr_list or first is None:
                        raise RuntimeError("No result produced for worst-case overlay.")

                    thr_worst: np.ndarray = np.nanmin(np.stack(thr_list, axis=0), axis=0)

                    h, w = thr_worst.shape
                    finite = np.isfinite(thr_worst)
                    img = np.zeros((h, w, 4), dtype=np.uint8)
                    if np.any(finite):
                        vmin = float(np.nanpercentile(thr_worst[finite], 2))
                        vmax = float(np.nanpercentile(thr_worst[finite], 98))
                        if vmin >= vmax:
                            vmax = vmin + 1e-6
                        scaled = np.zeros_like(thr_worst, dtype=float)
                        scaled[finite] = np.clip(
                            (thr_worst[finite] - vmin) / (vmax - vmin), 0.0, 1.0
                        )
                        img[..., 0] = (scaled * 255).astype(np.uint8)
                        img[..., 1] = np.minimum(255, (scaled * 255 + 100)).astype(np.uint8)
                        img[..., 2] = (255 - (scaled * 255)).astype(np.uint8)
                        img[..., 3] = np.where(finite, 200, 0).astype(np.uint8)
                    else:
                        img[..., 0] = 255
                        img[..., 3] = 120

                    assert first is not None
                    d_grid = first.get("d_grid_m")
                    if isinstance(d_grid, np.ndarray):
                        meets = thr_worst >= target_mbps
                        max_dist_m = (
                            float(np.nanmax(np.where(meets, d_grid, np.nan)))
                            if np.any(meets)
                            else 0.0
                        )
                        max_dist_km = max_dist_m / 1000.0
                    else:
                        max_dist_km = 0.0

                    if st.session_state.get("save_arrays"):
                        dump_arrays(
                            run_id, data_root / "debug", {"thr_worst": thr_worst.astype(np.float32)}
                        )

                    res = {
                        "image": img,
                        "bounds": first["bounds"],
                        "tx_lat": float(first["tx_lat"]),
                        "tx_lon": float(first["tx_lon"]),
                        "geotiff": first["geotiff"],
                        "csv": first["csv"],
                        "max_dist_km": float(max_dist_km),
                    }

                logger.info("[RUN %s] done dt=%.2fs", run_id, time.time() - t0)
                log_resources("post_run")

                st.session_state["res"] = res
                status.update(label="Done", state="complete")
                st.success("Simulation finished.")
    except Exception as e:
        logger.exception("[RUN %s] Simulation failed", run_id)
        st.error(f"Simulation failed: {e}")
    finally:
        st.session_state["run_sim"] = False
        st.session_state["_sim_running"] = False

# ------------------- Render result -------------------
res_obj = st.session_state.get("res")
if not isinstance(res_obj, dict):
    st.info("Set parameters and click Run Simulation to generate a heatmap.")
elif res_obj.get("image") is None:
    st.warning("No finite throughput to render. Adjust parameters and run again.")
else:
    res_dict = res_obj
    if "base_map" in globals() and isinstance(base_map, folium.Map) and base_map is not None:
        map_obj = base_map
    else:
        map_obj = folium.Map(
            location=[float(res_dict["tx_lat"]), float(res_dict["tx_lon"])],
            zoom_start=12,
            control_scale=True,
            tiles="OpenStreetMap",
        )
        folium.Marker(
            [float(res_dict["tx_lat"]), float(res_dict["tx_lon"])],
            tooltip=f"TX: {float(res_dict['tx_lat']):.6f}, {float(res_dict['tx_lon']):.6f}",
            icon=folium.Icon(color="blue", icon="wifi", prefix="fa"),
        ).add_to(map_obj)
        folium.Circle(
            [float(res_dict["tx_lat"]), float(res_dict["tx_lon"])],
            radius=float(aoi_radius_km) * 1000.0,
            color="#3388ff",
            weight=1,
            fill=False,
        ).add_to(map_obj)

    folium.raster_layers.ImageOverlay(
        image=res_dict["image"],
        bounds=res_dict["bounds"],
        opacity=0.75,
        interactive=False,
        cross_origin=False,
        zindex=3,
    ).add_to(map_obj)

    los_mask_obj = st.session_state.get("_last_los_mask")
    if st.session_state.get("show_los_mask") and isinstance(los_mask_obj, np.ndarray):
        los = los_mask_obj
        h2, w2 = los.shape
        los_img = np.zeros((h2, w2, 4), dtype=np.uint8)
        los_img[..., 1] = np.where(los, 255, 0)
        los_img[..., 3] = np.where(los, 60, 0)
        folium.raster_layers.ImageOverlay(
            image=los_img,
            bounds=res_dict["bounds"],
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=4,
        ).add_to(map_obj)

    folium.CircleMarker(
        location=[float(res_dict["tx_lat"]), float(res_dict["tx_lon"])],
        radius=6,
        color="red",
        fill=True,
        fill_opacity=1,
        popup="TX",
    ).add_to(map_obj)

    st_folium(map_obj, height=600, use_container_width=True, key="sim_map")
    st.success(f"Max distance for ≥ {target_mbps} Mbps: {float(res_dict['max_dist_km']):.2f} km")
    st.caption(f"Saved: GeoTIFF → {res_dict['geotiff']}  |  CSV → {res_dict['csv']}")
    st.caption("Throughput scale uses 0–98th percentile for color normalization.")
