"""
Chunked regionprops passes with incremental accumulators.

RAM contract
------------
Morphology pass : dict accumulator — one entry per cell, updated in-place as
                  chunks arrive; no intermediate DataFrames.
Intensity pass  : pre-allocated numpy arrays of length (max_label + 1) —
                  one per intensity property plus one for best-area tracking;
                  updated in-place per chunk; columns added to the output
                  DataFrame at the end of each channel.

Neither pass materialises more than ``n_jobs`` chunks at once.
"""

import sys

import h5py
import joblib
import numpy as np
import pandas as pd
import skimage.measure
import tifffile
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import zarr

from ._chunks import iter_chunk_coords
from ._props import EXTRA_PROPS, MORPHOLOGY_PROPS, NAME_MAP, TAIL_COLS


# ---------------------------------------------------------------------------- #
#                               Worker functions                               #
# ---------------------------------------------------------------------------- #

def _morph_worker(
    mask_zarr_dir: str,
    rrs: int, rre: int, ccs: int, cce: int,
    props: tuple[str, ...],
) -> tuple[dict, int, int]:
    z = zarr.open_array(mask_zarr_dir, mode="r")
    chunk = np.asarray(z[rrs:rre, ccs:cce])
    result = skimage.measure.regionprops_table(chunk, properties=props)
    # Return raw dict + offsets; centroid correction applied in the main process
    return result, rrs, ccs


def _intensity_worker_tiled(
    mask_zarr_dir: str,
    img_path: str,
    channel_idx: int,
    rrs: int, rre: int, ccs: int, cce: int,
    builtin_props: tuple[str, ...],
    extra_prop_names: tuple[str, ...],
) -> dict:
    mask_z = zarr.open_array(mask_zarr_dir, mode="r")
    img_z = zarr.open(tifffile.imread(img_path, aszarr=True, level=0), mode="r")

    chunk_mask = np.asarray(mask_z[rrs:rre, ccs:cce])
    # OME-TIFF: shape is (C, H, W)
    chunk_img = np.asarray(img_z[channel_idx, rrs:rre, ccs:cce])

    extra_fns = tuple(EXTRA_PROPS[n] for n in extra_prop_names)
    return skimage.measure.regionprops_table(
        chunk_mask, chunk_img,
        properties=builtin_props,
        extra_properties=extra_fns,
    )


def _intensity_worker_nontiled(
    mask_zarr_dir: str,
    full_channel: np.ndarray,
    rrs: int, rre: int, ccs: int, cce: int,
    builtin_props: tuple[str, ...],
    extra_prop_names: tuple[str, ...],
) -> dict:
    """Used for non-tiled TIFFs and HDF5 (full channel already in RAM)."""
    mask_z = zarr.open_array(mask_zarr_dir, mode="r")
    chunk_mask = np.asarray(mask_z[rrs:rre, ccs:cce])
    chunk_img = full_channel[rrs:rre, ccs:cce]

    extra_fns = tuple(EXTRA_PROPS[n] for n in extra_prop_names)
    return skimage.measure.regionprops_table(
        chunk_mask, chunk_img,
        properties=builtin_props,
        extra_properties=extra_fns,
    )


# ---------------------------------------------------------------------------- #
#                             Morphology pass                                  #
# ---------------------------------------------------------------------------- #

def morphology_pass(
    mask_zarr_dir: str,
    shape: tuple[int, int],
    chunk_size: int,
    overlap: int,
    extra_mask_props: list[str] | None,
    n_jobs: int,
) -> pd.DataFrame:
    """
    Compute morphological properties for all cells in the mask.

    Returns a DataFrame with columns matching NAME_MAP output names,
    CellID as the first column.  No DataFrame is built until the end.
    """
    props = list(MORPHOLOGY_PROPS)
    if extra_mask_props:
        props = list(dict.fromkeys(props + extra_mask_props))  # dedup, preserve order

    coords = list(iter_chunk_coords(shape, chunk_size, overlap))

    gen = joblib.Parallel(backend="loky", n_jobs=n_jobs, return_as="generator")(
        joblib.delayed(_morph_worker)(mask_zarr_dir, rrs, rre, ccs, cce, tuple(props))
        for rrs, rre, ccs, cce in coords
    )

    # Dict accumulator: label -> property dict (largest area seen wins)
    acc: dict[int, dict] = {}

    with logging_redirect_tqdm():
        for chunk_result, rrs, ccs in tqdm.tqdm(
            gen, total=len(coords), desc="morphology",
            disable=not sys.stderr.isatty(),
        ):
            labels = chunk_result["label"]
            areas = chunk_result["area"]
            n = len(labels)
            if n == 0:
                continue
            for i in range(n):
                lbl = int(labels[i])
                area = int(areas[i])
                if lbl == 0:
                    continue
                existing = acc.get(lbl)
                if existing is None or area > existing["area"]:
                    row = {k: chunk_result[k][i] for k in chunk_result}
                    # Apply centroid offset to convert chunk-local → global coords
                    row["centroid-0"] = row["centroid-0"] + rrs
                    row["centroid-1"] = row["centroid-1"] + ccs
                    acc[lbl] = row

    if not acc:
        raise RuntimeError("No labelled cells found in mask.")

    # Build DataFrame in one allocation from numpy arrays
    keys = list(next(iter(acc.values())).keys())
    arrays = {k: np.array([acc[lbl][k] for lbl in sorted(acc)]) for k in keys}

    # Rename and reorder columns
    df = pd.DataFrame(
        {NAME_MAP.get(k, k): arrays[k] for k in keys},
    )
    # CellID must come from 'label' key
    df = _reorder_columns(df)
    return df


# ---------------------------------------------------------------------------- #
#                             Intensity pass                                   #
# ---------------------------------------------------------------------------- #

def intensity_pass(
    mask_zarr_dir: str,
    img_path: str,
    img_format: str,
    hdf5_key: str | None,
    shape: tuple[int, int],
    channel_idx: int,
    channel_name: str,
    chunk_size: int,
    overlap: int,
    valid_labels: np.ndarray,
    max_label: int,
    intensity_props: list[str] | None,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    """
    Quantify one image channel against the mask.

    Returns a dict mapping column name -> 1-D numpy array of length
    ``len(valid_labels)``, in the same order as ``valid_labels``.
    """
    # Determine which props to compute
    _props = list(intensity_props) if intensity_props else []
    builtin_props = tuple(
        p for p in ["label", "area", "intensity_mean"] + _props
        if p not in EXTRA_PROPS
    )
    extra_prop_names = tuple(p for p in _props if p in EXTRA_PROPS)

    # Pre-allocate accumulators indexed by label
    best_area = np.zeros(max_label + 1, dtype=np.int64)
    # Map prop name -> accumulator array
    # intensity_mean is always present; extras are additional
    prop_keys = ["intensity_mean"] + list(extra_prop_names)
    accs: dict[str, np.ndarray] = {
        k: np.full(max_label + 1, np.nan, dtype=np.float64) for k in prop_keys
    }

    coords = list(iter_chunk_coords(shape, chunk_size, overlap))

    if img_format == "tiff_tiled":
        gen = joblib.Parallel(backend="loky", n_jobs=n_jobs, return_as="generator")(
            joblib.delayed(_intensity_worker_tiled)(
                mask_zarr_dir, img_path, channel_idx,
                rrs, rre, ccs, cce,
                builtin_props, extra_prop_names,
            )
            for rrs, rre, ccs, cce in coords
        )
    else:
        # Non-tiled TIFF or HDF5: load full channel once, slice per chunk in threads
        full_channel = _load_full_channel(img_path, img_format, hdf5_key, channel_idx)
        gen = joblib.Parallel(backend="threading", n_jobs=n_jobs, return_as="generator")(
            joblib.delayed(_intensity_worker_nontiled)(
                mask_zarr_dir, full_channel,
                rrs, rre, ccs, cce,
                builtin_props, extra_prop_names,
            )
            for rrs, rre, ccs, cce in coords
        )

    with logging_redirect_tqdm():
        for chunk_result in tqdm.tqdm(
            gen, total=len(coords), desc=f"  {channel_name}", leave=False,
            disable=not sys.stderr.isatty(),
        ):
            labels = chunk_result["label"]
            areas = chunk_result["area"]
            if len(labels) == 0:
                continue

            # Clip to pre-allocated range (labels outside max_label are background/artifacts)
            in_range = labels <= max_label
            labels = labels[in_range]
            areas = areas[in_range]

            better = areas > best_area[labels]
            win_labels = labels[better]
            if len(win_labels) == 0:
                continue

            best_area[win_labels] = areas[better]
            for key in prop_keys:
                # Extra props may generate columns like 'gini_index' directly
                src_key = key if key in chunk_result else _find_extra_key(chunk_result, key)
                if src_key is not None:
                    accs[key][win_labels] = chunk_result[src_key][in_range][better]

    # Slice to valid labels only
    result: dict[str, np.ndarray] = {}
    for key in prop_keys:
        col_name = channel_name if key == "intensity_mean" else f"{channel_name}_{key}"
        result[col_name] = accs[key][valid_labels]

    return result


def _find_extra_key(chunk_result: dict, prop_name: str) -> str | None:
    """Find the actual key in chunk_result for an extra property (handles skimage suffixes)."""
    if prop_name in chunk_result:
        return prop_name
    # skimage may name it directly as the function name
    for k in chunk_result:
        if k == prop_name or k.startswith(prop_name):
            return k
    return None


def _load_full_channel(
    img_path: str, img_format: str, hdf5_key: str | None, channel_idx: int
) -> np.ndarray:
    if img_format == "hdf5":
        with h5py.File(img_path, "r") as f:
            return np.asarray(f[hdf5_key][0, :, :, channel_idx])
    else:
        # Non-tiled TIFF: tifffile reads the z-th page
        return tifffile.imread(img_path, key=channel_idx)


# ---------------------------------------------------------------------------- #
#                            Column ordering                                   #
# ---------------------------------------------------------------------------- #

def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place CellID first, morphology tail last, intensity columns in the middle.
    Matches ori's histoCAT convention.
    """
    tail = [c for c in TAIL_COLS if c in df.columns]
    middle = [c for c in df.columns if c not in ("CellID",) + TAIL_COLS]
    ordered = ["CellID"] + middle + tail
    return df[ordered]


def build_output(
    morph_df: pd.DataFrame,
    channel_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Combine morphology DataFrame with per-channel intensity arrays.

    ``channel_arrays`` maps column_name -> 1-D array of length len(valid_labels).
    All arrays are assembled into a single new DataFrame (one allocation).
    """
    # Collect all column data as numpy arrays to avoid CoW copies
    all_arrays: dict[str, np.ndarray] = {}

    # Intensity columns first (will be reordered later)
    for col, arr in channel_arrays.items():
        all_arrays[col] = arr

    # Morphology columns
    for col in morph_df.columns:
        all_arrays[col] = morph_df[col].to_numpy()

    result = pd.DataFrame(all_arrays)
    return _reorder_columns(result)
