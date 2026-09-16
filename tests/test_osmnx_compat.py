from __future__ import annotations

import ast
import logging
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast


SOURCE = Path(__file__).resolve().parents[1] / "manet_heatmap_appV23.py"

FUNCTION_NAMES = {
    "configure_osmnx",
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
