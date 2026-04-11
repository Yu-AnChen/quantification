"""
Chunk coordinate computation, overlap estimation, and mask zarr conversion.
"""
import os
from typing import Iterator

import joblib
import numpy as np
import skimage.measure
import skimage.segmentation
import tifffile
import tqdm
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


def mask_to_zarr(mask: np.ndarray, chunk_size: int, store_path: str) -> None:
    """
    Write a 2-D integer mask to a zarr DirectoryStore.
    Each worker will re-open this store independently.
    """
    z = zarr.open_array(
        store_path,
        mode="w",
        shape=mask.shape,
        dtype=mask.dtype,
        chunks=(chunk_size, chunk_size),
    )
    z[:] = mask


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
            max_row[min_row == 0],          # touching top:    extent downward
            max_col[min_col == 0],          # touching left:   extent rightward
            h - min_row[max_row == h],      # touching bottom: extent upward
            w - min_col[max_col == w],      # touching right:  extent leftward
        ]
    )
    return distances if len(distances) else np.array([0], dtype=np.int64)


def _edge_worker(mask_zarr_dir: str, rrs: int, rre: int, ccs: int, cce: int) -> np.ndarray:
    z = zarr.open_array(mask_zarr_dir, mode="r")
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
    ``_OVERLAP_ALIGN``.
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
    for distances in tqdm.tqdm(gen, total=len(coords), desc="computing overlap", leave=False):
        d = int(np.max(distances))
        if d > max_dist:
            max_dist = d

    overlap = int(_OVERLAP_ALIGN * np.ceil(max_dist / _OVERLAP_ALIGN))
    return overlap
