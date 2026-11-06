"""Geographic utilities.

- haversine_km: works on numpy AND torch (torch version differentiable).
- latlon_to_unit / unit_to_latlon: 3D unit-vector conversions.
- country_of: batched reverse geocoding -> alpha-2 codes.
- continent_of: alpha-2 -> continent name (graceful fallback).

Conventions: inputs in DEGREES; Earth radius 6371 km. Guards against NaNs
from floating point in asin/acos.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------
# Haversine
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts numpy arrays/scalars or torch tensors.

    Inputs are in degrees. Differentiable when given torch tensors.
    """
    is_torch = _is_torch(lat1, lon1, lat2, lon2)
    if is_torch:
        import torch

        rad = math.pi / 180.0
        lat1r, lon1r, lat2r, lon2r = (x * rad for x in (lat1, lon1, lat2, lon2))
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2
        )
        a = torch.clamp(a, 0.0, 1.0)
        c = 2 * torch.asin(torch.sqrt(a + 1e-16))
        return EARTH_RADIUS_KM * c

    lat1r, lon1r, lat2r, lon2r = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def _is_torch(*args) -> bool:
    try:
        import torch

        return any(isinstance(a, torch.Tensor) for a in args)
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Unit-vector conversions (numpy)
# --------------------------------------------------------------------------
def latlon_to_unit(lat, lon):
    """(lat, lon) in degrees -> (..., 3) unit vectors."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def unit_to_latlon(xyz):
    """(..., 3) vectors -> (lat, lon) in degrees. Normalizes internally."""
    xyz = np.asarray(xyz, dtype=np.float64)
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz = xyz / np.clip(norm, 1e-12, None)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def mean_latlon(lats, lons):
    """Mean location via mean of unit vectors (robust to antimeridian)."""
    v = latlon_to_unit(lats, lons).mean(axis=0)
    lat, lon = unit_to_latlon(v)
    return float(lat), float(lon)


# --------------------------------------------------------------------------
# Reverse geocoding + continents
# --------------------------------------------------------------------------
_RG = None


def _rg():
    global _RG
    if _RG is None:
        import reverse_geocoder as rg

        _RG = rg
    return _RG


def country_of(lats, lons) -> list[str]:
    """Batched reverse geocoding -> list of ISO alpha-2 country codes.

    Call once per array; reverse_geocoder is slow per-point.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    coords = list(zip(lats.tolist(), lons.tolist()))
    results = _rg().search(coords)
    return [r.get("cc", "") for r in results]


def continent_of(country_code: str) -> str:
    """alpha-2 country code -> continent name; 'Unknown' on failure."""
    if not country_code:
        return "Unknown"
    try:
        from pycountry_convert import (
            country_alpha2_to_continent_code,
            convert_continent_code_to_continent_name,
        )

        cont_code = country_alpha2_to_continent_code(country_code.upper())
        return convert_continent_code_to_continent_name(cont_code)
    except Exception:
        # Edge cases: AQ (Antarctica), TL, EH, etc.
        special = {
            "AQ": "Antarctica",
            "TL": "Asia",
            "EH": "Africa",
            "PN": "Oceania",
        }
        return special.get(country_code.upper(), "Unknown")


def continents_of(country_codes: Iterable[str]) -> list[str]:
    return [continent_of(c) for c in country_codes]
