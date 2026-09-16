from __future__ import annotations

import ast
import logging
import time
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from types import SimpleNamespace
from typing import cast


SOURCE = Path(__file__).resolve().parents[1] / "manet_heatmap_appV23.py"

FUNCTION_NAMES = {
    "configure_osmnx",
    "_fetch_osm_bbox_with_ui_cap",
    "_fetch_buildings_with_shrink",
    "get_osm_buildings_local_or_remote",
    "_osmnx_set_timeout",
    "_osmnx_fetch_buildings_core",
    "_osmnx_fetch_buildings_safe",
}


class _DummyResult:
    empty = False

    def __len__(self) -> int:
        return 1


def _load_osmnx_functions(fake_ox):
    """
    Load only the OSMnx compatibility helpers from the production source.

    The Streamlit application executes substantial code at module scope, so
    importing the complete module would start application initialization.
    AST extraction keeps these tests isolated and network-free while exercising
    the actual function bodies under test.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))

    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in FUNCTION_NAMES
    ]

    found = {node.name for node in selected}
    missing = FUNCTION_NAMES - found
    if missing:
        raise AssertionError(f"Expected production functions not found: {sorted(missing)}")

    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    namespace = {
        "__name__": "_manet_osmnx_compat_test_extract",
        "__file__": str(SOURCE),
        "APP_NAME": "manet_heatmap",
        "SESSION_ID": "test-session",
        "Path": Path,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "FuturesTimeout": FuturesTimeout,
        "ox": fake_ox,
        "gpd": SimpleNamespace(GeoDataFrame=object),
        "Callable": Callable,
        "cast": cast,
        "_safe_twrite": lambda *args, **kwargs: None,
        "logger": logging.getLogger("test.osmnx.compat"),
    }

    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class OSMnxCompatibilityTests(unittest.TestCase):
    def test_configure_osmnx_avoids_duplicate_timeout_kwarg(self):
        settings = SimpleNamespace(
            use_cache=False,
            cache_folder="",
            log_console=True,
            requests_timeout=180,
            requests_kwargs={
                "timeout": 55,
                "verify": False,
                "headers": {"X-Test": "1"},
            },
            default_user_agent="OSMnx default",
            max_query_area_size=1,
        )

        fake_ox = SimpleNamespace(
            __version__="2.1.1",
            settings=settings,
        )
        ns = _load_osmnx_functions(fake_ox)

        ns["configure_osmnx"](logging.getLogger("test.osmnx.configure"))

        self.assertEqual(settings.requests_timeout, (120, 120))
        self.assertNotIn("timeout", settings.requests_kwargs)
        self.assertFalse(settings.requests_kwargs["verify"])

    def test_background_osm_worker_is_streamlit_free(self):
        src = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(SOURCE))

        worker_functions = {
            "fetch_osm_buildings_bbox",
            "_osmnx_fetch_buildings_safe",
        }

        found = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in worker_functions:
                found[node.name] = node

        self.assertEqual(
            set(found),
            worker_functions,
            "Expected background OSM worker functions were not found",
        )

        violations = []

        for fn_name, fn_node in found.items():
            for node in ast.walk(fn_node):
                if isinstance(node, ast.Name) and node.id == "st":
                    violations.append(
                        f"{fn_name}: direct Streamlit reference at line {node.lineno}"
                    )

                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_safe_twrite"
                ):
                    violations.append(
                        f"{fn_name}: _safe_twrite call at line {node.lineno}"
                    )

        self.assertEqual(
            violations,
            [],
            "Background OSM worker path still depends on Streamlit context:\n"
            + "\n".join(violations),
        )

    def test_inner_overpass_fetch_has_no_nested_executor(self):
        src = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(SOURCE))

        target = None
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_osmnx_fetch_buildings_safe"
            ):
                target = node
                break

        self.assertIsNotNone(
            target,
            "_osmnx_fetch_buildings_safe not found in production source",
        )

        executor_refs = [
            node.lineno
            for node in ast.walk(target)
            if isinstance(node, ast.Name)
            and node.id == "ThreadPoolExecutor"
        ]

        self.assertEqual(
            executor_refs,
            [],
            "Inner OSM fetch still creates a nested ThreadPoolExecutor",
        )

        direct_core_calls = [
            node
            for node in ast.walk(target)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_osmnx_fetch_buildings_core"
        ]

        self.assertGreaterEqual(
            len(direct_core_calls),
            1,
            "_osmnx_fetch_buildings_safe must call the core fetch synchronously",
        )


    def test_osm_ui_cap_enforces_single_flight_across_reruns(self):
        class _EmptyResult:
            empty = True

            def __len__(self):
                return 0

        submissions = []
        shared_future = None

        class _StillRunningFuture:
            def result(self, timeout=None):
                raise FuturesTimeout()

            def cancel(self):
                # Simulate a worker that has already started and therefore
                # cannot be cancelled by Future.cancel().
                return False

            def done(self):
                return False

        shared_future = _StillRunningFuture()

        class _CountingExecutor:
            def __init__(self, max_workers=1):
                self.max_workers = max_workers

            def submit(self, fn, *args, **kwargs):
                submissions.append((fn, args, kwargs))
                return shared_future

            def shutdown(self, wait=True, cancel_futures=False):
                return None

        fake_ox = SimpleNamespace(settings=SimpleNamespace())
        ns = _load_osmnx_functions(fake_ox)

        ns["ThreadPoolExecutor"] = _CountingExecutor
        ns["_safe_twrite"] = lambda *args, **kwargs: None
        ns["fetch_osm_buildings_bbox"] = lambda *args, **kwargs: _EmptyResult()
        ns["gpd"] = SimpleNamespace(
            GeoDataFrame=lambda *args, **kwargs: _EmptyResult()
        )
        ns["st"] = SimpleNamespace(session_state={})

        bbox = (59.41, 59.46, 24.70, 24.81)

        first = ns["_fetch_osm_bbox_with_ui_cap"](bbox, 1.0)
        second = ns["_fetch_osm_bbox_with_ui_cap"](bbox, 1.0)

        self.assertTrue(first.empty)
        self.assertTrue(second.empty)

        self.assertEqual(
            len(submissions),
            1,
            "A second OSM worker was submitted while the first was still in-flight",
        )

    def test_osm_ui_wait_cap_bounds_wall_clock_time(self):
        class _EmptyGDF:
            empty = True

        fake_ox = SimpleNamespace(settings=SimpleNamespace())
        ns = _load_osmnx_functions(fake_ox)
        ns["st"] = SimpleNamespace(session_state={})

        def slow_fetch(lat_s, lat_n, lon_w, lon_e):
            time.sleep(1.6)
            return _EmptyGDF()

        # Replace all external/network behavior with deterministic local fakes.
        ns["fetch_osm_buildings_bbox"] = slow_fetch
        ns["gpd"] = SimpleNamespace(
            GeoDataFrame=lambda *args, **kwargs: _EmptyGDF()
        )

        t0 = time.perf_counter()
        result = ns["_fetch_osm_bbox_with_ui_cap"](
            bbox=(59.41, 59.46, 24.70, 24.81),
            ui_wait_s=1.0,
        )
        elapsed = time.perf_counter() - t0

        self.assertTrue(result.empty)

        # Allow scheduling tolerance, but the caller must return materially
        # before the 1.6 s worker itself finishes.
        self.assertLess(
            elapsed,
            1.30,
            f"UI wait cap did not bound wall-clock time: {elapsed:.3f}s",
        )

    def test_remote_orchestration_uses_ui_capped_fetch(self):
        bbox = (59.41, 59.46, 24.70, 24.81)

        fake_ox = SimpleNamespace(settings=SimpleNamespace())
        ns = _load_osmnx_functions(fake_ox)

        raw_calls = []
        bounded_calls = []

        raw_result = _DummyResult()
        bounded_result = _DummyResult()

        def raw_fetch(lat_s, lat_n, lon_w, lon_e):
            raw_calls.append((lat_s, lat_n, lon_w, lon_e))
            return raw_result

        def bounded_fetch(actual_bbox, ui_wait_s):
            bounded_calls.append((actual_bbox, ui_wait_s))
            return bounded_result

        ns["fetch_osm_buildings_bbox"] = raw_fetch
        ns["_fetch_osm_bbox_with_ui_cap"] = bounded_fetch
        ns["_osm_local_path"] = lambda *args, **kwargs: Path("unused.gpkg")
        ns["_safe_twrite"] = lambda *args, **kwargs: None
        ns["st"] = SimpleNamespace(
            session_state={"osm_ui_wait_cap_s": 7.0}
        )
        ns["logger"] = logging.getLogger("test.osmnx.remote")

        result = ns["get_osm_buildings_local_or_remote"](
            city_label="Tallinn",
            bbox=bbox,
            radius_km=3.0,
            source_mode="Overpass only",
            save_after_fetch=False,
        )

        self.assertEqual(
            raw_calls,
            [],
            "Remote orchestration bypassed the UI-capped fetch helper",
        )
        self.assertEqual(
            bounded_calls,
            [(bbox, 7.0)],
            "Remote orchestration did not use the configured UI wait cap",
        )
        self.assertIs(result, bounded_result)

    def test_shrink_retry_uses_ui_capped_fetch(self):
        class _EmptyResult:
            empty = True

            def __len__(self):
                return 0

        bbox = (59.41, 59.46, 24.70, 24.81)
        inner_bbox = (59.415, 59.455, 24.71, 24.80)

        fake_ox = SimpleNamespace(settings=SimpleNamespace())
        ns = _load_osmnx_functions(fake_ox)

        raw_calls = []
        bounded_calls = []

        bounded_result = _DummyResult()

        def raw_fetch(lat_s, lat_n, lon_w, lon_e):
            raw_calls.append((lat_s, lat_n, lon_w, lon_e))
            return bounded_result

        def bounded_fetch(actual_bbox, ui_wait_s):
            bounded_calls.append((actual_bbox, ui_wait_s))
            return bounded_result

        ns["get_osm_buildings_local_or_remote"] = (
            lambda *args, **kwargs: _EmptyResult()
        )
        ns["_shrink_bbox"] = lambda actual_bbox, ratio: inner_bbox
        ns["fetch_osm_buildings_bbox"] = raw_fetch
        ns["_fetch_osm_bbox_with_ui_cap"] = bounded_fetch
        ns["_safe_twrite"] = lambda *args, **kwargs: None
        ns["gpd"] = SimpleNamespace(
            GeoDataFrame=lambda *args, **kwargs: _EmptyResult()
        )
        ns["st"] = SimpleNamespace(
            session_state={"osm_ui_wait_cap_s": 7.0},
            caption=lambda *args, **kwargs: None,
        )

        result = ns["_fetch_buildings_with_shrink"](
            bbox=bbox,
            radius_km=1.0,
            auto_shrink=True,
            max_aoi_km=3.0,
            source_mode="Overpass only",
            save_after_fetch=False,
            city_label="Tallinn",
        )

        self.assertEqual(
            raw_calls,
            [],
            "Shrink retry bypassed the UI-capped fetch helper",
        )
        self.assertEqual(
            bounded_calls,
            [(inner_bbox, 7.0)],
            "Shrink retry did not use the configured UI wait cap",
        )
        self.assertIs(result, bounded_result)

    def test_features_from_bbox_uses_v2_bbox_contract_directly(self):
        calls = []
        result = _DummyResult()
        tags = {"building": True}

        def features_from_bbox(*args):
            calls.append(args)

            # Simulate the installed OSMnx 2.x signature:
            # features_from_bbox(bbox, tags)
            if len(args) != 2:
                raise TypeError("expected bbox and tags")

            return result

        fake_ox = SimpleNamespace(
            settings=SimpleNamespace(),
            features_from_bbox=features_from_bbox,
        )

        ns = _load_osmnx_functions(fake_ox)

        returned = ns["_osmnx_fetch_buildings_core"](
            north=52.20,
            south=52.10,
            east=5.20,
            west=5.10,
            tags=tags,
        )

        self.assertIs(returned, result)

        # OSMnx 2.x bbox contract is:
        # (left, bottom, right, top) == (west, south, east, north)
        self.assertEqual(
            calls,
            [
                (
                    (5.10, 52.10, 5.20, 52.20),
                    tags,
                )
            ],
        )

    def test_timeout_uses_requests_timeout_without_duplicate_kwarg(self):
        settings = SimpleNamespace(
            requests_timeout=180,
            requests_kwargs={},
            overpass_settings="[out:json][timeout:{timeout}]{maxsize}",
            overpass_rate_limit=True,
            overpass_url="https://overpass-api.de/api",
        )

        fake_ox = SimpleNamespace(settings=settings)
        ns = _load_osmnx_functions(fake_ox)

        ns["_osmnx_set_timeout"](27)

        self.assertEqual(settings.requests_timeout, 27)
        self.assertNotIn("timeout", settings.requests_kwargs)

    def test_mirror_rotation_sets_normalized_overpass_base_url(self):
        settings = SimpleNamespace(
            requests_timeout=180,
            requests_kwargs={},
            overpass_settings="[out:json][timeout:{timeout}]{maxsize}",
            overpass_rate_limit=True,
            overpass_url="https://overpass-api.de/api",
        )

        fake_ox = SimpleNamespace(settings=settings)
        ns = _load_osmnx_functions(fake_ox)

        # Prevent all network activity. We only want to verify how the wrapper
        # configures OSMnx before invoking the core fetch.
        ns["_osmnx_fetch_buildings_core"] = (
            lambda north, south, east, west, tags: _DummyResult()
        )

        endpoint = "https://lz4.overpass-api.de/api/interpreter"

        result, working_endpoint = ns["_osmnx_fetch_buildings_safe"](
            north=52.20,
            south=52.10,
            east=5.20,
            west=5.10,
            tags={"building": True},
            timeout_s=11,
            endpoints=[endpoint],
        )

        self.assertFalse(result.empty)
        self.assertEqual(working_endpoint, endpoint)

        # OSMnx appends "/interpreter" itself, so overpass_url must contain
        # the base URL rather than the final interpreter endpoint.
        self.assertEqual(
            settings.overpass_url,
            "https://lz4.overpass-api.de/api",
        )

        self.assertEqual(settings.requests_timeout, 11)
        self.assertNotIn("timeout", settings.requests_kwargs)


if __name__ == "__main__":
    unittest.main()
