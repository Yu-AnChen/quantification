"""
Property definitions, custom extra properties, validation, and precision rounding.
"""

import numpy as np
import pandas as pd
import skimage.measure._regionprops as _rp


# ---------------------------------------------------------------------------- #
#                              Column name mapping                             #
# ---------------------------------------------------------------------------- #

# skimage regionprops key -> output CSV column name
NAME_MAP = {
    "label": "CellID",
    "centroid-0": "Y_centroid",
    "centroid-1": "X_centroid",
    "area": "Area",
    "major_axis_length": "MajorAxisLength",
    "minor_axis_length": "MinorAxisLength",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "extent": "Extent",
    "orientation": "Orientation",
}

# Morphology columns that go at the end of the CSV (histoCAT convention)
TAIL_COLS = (
    "X_centroid",
    "Y_centroid",
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "Solidity",
    "Extent",
    "Orientation",
)

# Default morphology properties requested from regionprops_table (skimage names)
MORPHOLOGY_PROPS = (
    "label",
    "centroid",
    "area",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "extent",
    "orientation",
)


# ---------------------------------------------------------------------------- #
#                              Precision rounding                              #
# ---------------------------------------------------------------------------- #

# None = cast to int; int = round to that many decimal places
_COL_PRECISION: dict[str, int | None] = {
    "CellID": None,
    "Area": None,
    "X_centroid": 2,
    "Y_centroid": 2,
    "MajorAxisLength": 4,
    "MinorAxisLength": 4,
    "Eccentricity": 4,
    "Solidity": 4,
    "Extent": 4,
    "Orientation": 4,
}
_DEFAULT_INTENSITY_PRECISION = 4


def apply_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Round all columns to their defined precision; build a new DataFrame (one allocation)."""
    arrays = {}
    for col in df.columns:
        prec = _COL_PRECISION.get(col, _DEFAULT_INTENSITY_PRECISION)
        arr = df[col].to_numpy()
        if prec is None:
            arrays[col] = arr.astype(np.int64)
        else:
            arrays[col] = np.round(arr, prec)
    return pd.DataFrame(arrays)


# ---------------------------------------------------------------------------- #
#                           Custom extra properties                            #
# ---------------------------------------------------------------------------- #


def gini_index(mask, intensity):
    x = intensity[mask].astype(float)
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x)
    return (n + 1 - 2 * cumx.sum() / cumx[-1]) / n


def intensity_median(mask, intensity):
    return np.median(intensity[mask])


def intensity_sum(mask, intensity):
    return float(np.sum(intensity[mask]))


# Name -> callable; used for validation and dispatch
EXTRA_PROPS: dict[str, callable] = {
    "gini_index": gini_index,
    "intensity_median": intensity_median,
    "intensity_sum": intensity_sum,
}


# ---------------------------------------------------------------------------- #
#                                 Validation                                   #
# ---------------------------------------------------------------------------- #

try:
    _BUILTIN_PROPS: frozenset[str] = frozenset(_rp.PROP_VALS)
except AttributeError:
    # Fallback for unexpected skimage versions
    _BUILTIN_PROPS = frozenset(
        p for p in dir(_rp.RegionProperties) if not p.startswith("_")
    )


def validate_props(props: list[str] | None) -> None:
    """Raise ValueError if any prop name is not a known builtin or extra property."""
    if props is None:
        return
    valid = _BUILTIN_PROPS | frozenset(EXTRA_PROPS)
    for p in props:
        if p not in valid:
            raise ValueError(
                f"Unknown property '{p}'. "
                f"Extra properties available: {sorted(EXTRA_PROPS)}."
            )
