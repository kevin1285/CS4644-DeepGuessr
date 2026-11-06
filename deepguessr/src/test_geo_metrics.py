"""Unit tests for geo + metrics. Run: pytest deepguessr/src/test_geo_metrics.py"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from geo import (  # noqa: E402
    continent_of,
    country_of,
    haversine_km,
    latlon_to_unit,
    mean_latlon,
    unit_to_latlon,
)
from metrics import evaluate  # noqa: E402

# Known coordinates
NYC = (40.7128, -74.0060)
LONDON = (51.5074, -0.1278)
TOKYO = (35.6762, 139.6503)


def test_haversine_nyc_london():
    d = haversine_km(NYC[0], NYC[1], LONDON[0], LONDON[1])
    assert abs(float(d) - 5570) < 100  # ~5570 km


def test_haversine_zero():
    assert float(haversine_km(NYC[0], NYC[1], NYC[0], NYC[1])) < 1e-3


def test_haversine_antimeridian():
    # Points straddling +/-180 should be close, not ~half the globe.
    d = haversine_km(0.0, 179.9, 0.0, -179.9)
    assert float(d) < 50  # ~22 km


def test_haversine_torch_matches_numpy_and_differentiable():
    torch = pytest.importorskip("torch")
    lat1 = torch.tensor([NYC[0]], requires_grad=True)
    lon1 = torch.tensor([NYC[1]], requires_grad=True)
    lat2 = torch.tensor([LONDON[0]])
    lon2 = torch.tensor([LONDON[1]])
    d = haversine_km(lat1, lon1, lat2, lon2)
    d.sum().backward()
    assert lat1.grad is not None and not torch.isnan(lat1.grad).any()
    d_np = haversine_km(np.array(NYC[0]), np.array(NYC[1]),
                        np.array(LONDON[0]), np.array(LONDON[1]))
    assert abs(float(d.detach()) - float(d_np)) < 1.0


def test_unit_vector_roundtrip():
    lats = np.array([40.7, -33.9, 35.7, 0.0])
    lons = np.array([-74.0, 151.2, 139.7, 179.9])
    xyz = latlon_to_unit(lats, lons)
    lat2, lon2 = unit_to_latlon(xyz)
    assert np.allclose(lats, lat2, atol=1e-6)
    assert np.allclose(lons, lon2, atol=1e-6)


def test_mean_latlon_antimeridian():
    # Mean of two points near antimeridian should stay near +/-180, not 0.
    lat, lon = mean_latlon([0.0, 0.0], [179.0, -179.0])
    assert abs(lat) < 1e-6
    assert abs(abs(lon) - 180) < 2


def test_country_of_known_points():
    ccs = country_of([NYC[0], LONDON[0], TOKYO[0]],
                     [NYC[1], LONDON[1], TOKYO[1]])
    assert ccs[0] == "US"
    assert ccs[1] == "GB"
    assert ccs[2] == "JP"


def test_continent_of():
    assert continent_of("US") == "North America"
    assert continent_of("JP") == "Asia"
    assert continent_of("") == "Unknown"
    assert continent_of("AQ") == "Antarctica"


def test_evaluate_perfect_prediction():
    lat = np.array([40.7, 51.5, 35.7])
    lon = np.array([-74.0, -0.1, 139.7])
    m = evaluate(lat, lon, lat, lon)
    assert m["median_km"] < 1e-3
    assert m["acc@25km"] == 1.0
    assert m["country_acc"] == 1.0
    assert m["n"] == 3
