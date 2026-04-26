"""
Chunk coordinate computation, overlap estimation, and mask zarr conversion.
"""

import pathlib
import sys
from typing import Iterator
import warnings

import joblib
import numpy as np
import skimage.measure
import skimage.segmentation
import tifffile
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import zarr

_TARGET_CHUNK = 4096
_OVERLAP_ALIGN = 16  # overlap is rounded up to a multiple of this


def compute_chunk_size(tile_size: int | None, user_override: int | None = None) -> int:
    """
    Compute the chunk size to use for processing.

    If ``user_override`` is given it is used directly (must be divisible by 16).
    Otherwise the chunk size snaps to the nearest multiple of ``tile_size``
    that is closest to ``_TARGET_CHUNK``.  Falls back to ``_TARGET_CHUNK``
    when the image is not tiled.
    """
    if user_override is not None:
        if user_override % 16 != 0:
            raise ValueError(
                f"--chunk-size must be divisible by 16 (got {user_override})."
            )
        return user_override

    if tile_size is None:
        return _TARGET_CHUNK

    n_tiles = max(1, round(_TARGET_CHUNK / tile_size))
    return n_tiles * tile_size


def mask_to_zarr(mask_path: str, chunk_size: int, store_path: str) -> None:
    """
    Copy a 2-D integer mask TIFF into a zarr store using TiffFile.aszarr().

    Tiled TIFFs: iterates chunk by chunk (no overlap) via iter_chunk_coords so
    only chunk_size × chunk_size pixels are live in RAM per step; chunk_size is
    a multiple of tile_size so reads are tile-aligned.

    Non-tiled TIFFs: iterates full-width strips of chunk_size rows. Strips span
    the full image width so column sub-reads would re-read the same strip data
    repeatedly; full-width reads minimise I/O at the cost of higher RAM per step.
    """
    with tifffile.TiffFile(mask_path) as tf:
        src = zarr.open(tf.aszarr(), mode="r")
        if not isinstance(src, zarr.Array):
            src = src[str(0)]  # OME-TIFF group hierarchy — take first array
        h, w = int(src.shape[-2]), int(src.shape[-1])

        dst = zarr.open_array(
            pathlib.Path(store_path),
            mode="w",
            shape=(h, w),
            dtype=np.int32,
            chunks=(chunk_size, chunk_size),
        )

        is_tiled = tf.series[0].levels[0].pages[0].is_tiled
        with logging_redirect_tqdm():
            if is_tiled:
                coords = list(iter_chunk_coords((h, w), chunk_size, overlap=0))
                for rrs, rre, ccs, cce in tqdm.tqdm(
                    coords,
                    desc="mask → zarr",
                    leave=False,
                    disable=not sys.stderr.isatty(),
                ):
                    dst[rrs:rre, ccs:cce] = src[rrs:rre, ccs:cce]
            else:
                for rs in tqdm.tqdm(
                    range(0, h, chunk_size),
                    desc="mask → zarr",
                    leave=False,
                    disable=not sys.stderr.isatty(),
                ):
                    re = min(rs + chunk_size, h)
                    dst[rs:re, :] = src[rs:re, :]


def iter_chunk_coords(
    shape: tuple[int, int],
    chunk_size: int,
    overlap: int,
) -> Iterator[tuple[int, int, int, int]]:
    """
    Yield (rrs, rre, ccs, cce) for each overlapping chunk covering ``shape``.
    Coordinates are clipped to image bounds.
    """
    h, w = shape
    nr = int(np.ceil(h / chunk_size))
    nc = int(np.ceil(w / chunk_size))
    for ri in range(nr):
        for ci in range(nc):
            rrs = max(ri * chunk_size - overlap, 0)
            rre = min((ri + 1) * chunk_size + overlap, h)
            ccs = max(ci * chunk_size - overlap, 0)
            cce = min((ci + 1) * chunk_size + overlap, w)
            yield rrs, rre, ccs, cce


# ---------------------------------------------------------------------------- #
#                          Overlap estimation                                  #
# ---------------------------------------------------------------------------- #


def _edge_distances(mask_chunk: np.ndarray) -> np.ndarray:
    """
    For labels touching any edge of ``mask_chunk``, return the distance
    from the touched edge to the far side of the label's bounding box.
    This is the overlap a neighbouring chunk needs to see the full label.
    """
    h, w = mask_chunk.shape
    cleared = skimage.segmentation.clear_border(mask_chunk)
    edge_only = mask_chunk.copy()
    edge_only[cleared > 0] = 0
    if np.all(edge_only == 0):
        return np.array([0], dtype=np.int64)

    t = skimage.measure.regionprops_table(edge_only, properties=("bbox",))
    min_row = t["bbox-0"]
    min_col = t["bbox-1"]
    max_row = t["bbox-2"]
    max_col = t["bbox-3"]

    distances = np.concatenate(
        [
            max_row[min_row == 0],  # touching top:    extent downward
            max_col[min_col == 0],  # touching left:   extent rightward
            h - min_row[max_row == h],  # touching bottom: extent upward
            w - min_col[max_col == w],  # touching right:  extent leftward
        ]
    )
    return distances


def _edge_worker(
    mask_zarr_dir: str, rrs: int, rre: int, ccs: int, cce: int
) -> np.ndarray:
    z = zarr.open_array(pathlib.Path(mask_zarr_dir), mode="r")
    chunk = np.asarray(z[rrs:rre, ccs:cce])
    return _edge_distances(chunk)


def compute_overlap(
    mask_zarr_dir: str,
    shape: tuple[int, int],
    chunk_size: int,
    n_jobs: int,
) -> int:
    """
    Estimate the pixel overlap needed so that no cell is split across chunks.

    Processes non-overlapping blocks in parallel (threading, I/O-bound),
    then rounds the maximum distance up to the nearest multiple of
    ``_OVERLAP_ALIGN``.  The result is capped at ``chunk_size - _OVERLAP_ALIGN``
    so that overlap never reaches chunk_size (which would cause adjacent chunks
    to start at the same offset, defeating chunked processing).  A warning is
    emitted when the cap is applied; cells larger than the cap will appear in
    multiple chunks and the largest-area instance is used.
    """
    h, w = shape
    nr = int(np.ceil(h / chunk_size))
    nc = int(np.ceil(w / chunk_size))

    coords = [
        (ri * chunk_size, (ri + 1) * chunk_size, ci * chunk_size, (ci + 1) * chunk_size)
        for ri in range(nr)
        for ci in range(nc)
    ]

    gen = joblib.Parallel(backend="threading", n_jobs=n_jobs, return_as="generator")(
        joblib.delayed(_edge_worker)(mask_zarr_dir, rrs, rre, ccs, cce)
        for rrs, rre, ccs, cce in coords
    )

    max_dist = 0
    with logging_redirect_tqdm():
        for distances in tqdm.tqdm(
            gen,
            total=len(coords),
            desc="computing overlap",
            leave=False,
            disable=not sys.stderr.isatty(),
        ):
            d = int(np.max(distances))
            if d > max_dist:
                max_dist = d

    overlap = int(_OVERLAP_ALIGN * np.ceil(max_dist / _OVERLAP_ALIGN))
    cap = chunk_size - _OVERLAP_ALIGN
    if overlap > cap:
        warnings.warn(
            f"Largest border-touching cell requires {overlap} px overlap but "
            f"chunk_size={chunk_size}. Capping overlap at {cap} px; cells larger "
            "than the cap will appear in multiple chunks (largest-area instance used).",
            UserWarning,
            stacklevel=2,
        )
        overlap = cap
    return overlap
