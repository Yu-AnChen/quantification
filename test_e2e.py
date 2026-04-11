"""
Synthetic end-to-end test.

Creates a small tiled OME-TIFF image + mask, runs the pipeline, and checks:
  - All cells present
  - Centroids are in global (not chunk-local) coordinates
  - Column order matches ori convention
  - Intensity and morphology values are sensible
"""
import os
import sys
import tempfile
import pathlib

import numpy as np
import pandas as pd
import skimage.measure
import tifffile
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import chunkprop


# ---------------------------------------------------------------------------- #
#                              Fixtures / helpers                              #
# ---------------------------------------------------------------------------- #

def make_synthetic_data(tmp_path, n_cells=20, img_h=512, img_w=512, n_channels=3, tile_size=128):
    """
    Create a synthetic mask and multichannel image saved as tiled OME-TIFFs.
    Returns (mask_path, img_path, marker_csv_path, expected_centroids).
    """
    rng = np.random.default_rng(42)

    # --- Mask: place non-overlapping square cells ---
    mask = np.zeros((img_h, img_w), dtype=np.int32)
    cell_size = 20
    cell_id = 1
    positions = []
    for r in range(cell_size, img_h - cell_size, cell_size + 5):
        for c in range(cell_size, img_w - cell_size, cell_size + 5):
            if cell_id > n_cells:
                break
            mask[r : r + cell_size, c : c + cell_size] = cell_id
            positions.append((r + cell_size / 2, c + cell_size / 2))  # row, col centroid
            cell_id += 1
        if cell_id > n_cells:
            break
    actual_n_cells = cell_id - 1

    # --- Image: constant intensity per channel per cell ---
    img = np.zeros((n_channels, img_h, img_w), dtype=np.uint16)
    for ch in range(n_channels):
        for cid in range(1, actual_n_cells + 1):
            img[ch][mask == cid] = cid * 100 + ch * 10  # deterministic

    # --- Save files ---
    mask_path = str(tmp_path / "mask.ome.tif")
    img_path = str(tmp_path / "image.ome.tif")
    marker_csv_path = str(tmp_path / "markers.csv")

    options = {"tile": (tile_size, tile_size), "compression": "zlib"}
    tifffile.imwrite(mask_path, mask, **options)
    tifffile.imwrite(img_path, img, **options)

    pd.DataFrame({"marker_name": [f"CH{i}" for i in range(n_channels)]}).to_csv(
        marker_csv_path, index=False
    )

    expected_centroids = {i + 1: positions[i] for i in range(actual_n_cells)}
    return mask_path, img_path, marker_csv_path, expected_centroids, actual_n_cells


# ---------------------------------------------------------------------------- #
#                                    Tests                                     #
# ---------------------------------------------------------------------------- #

def test_all_cells_found(tmp_path):
    mask_path, img_path, markers, expected, n_cells = make_synthetic_data(tmp_path)
    out = str(tmp_path / "output")

    chunkprop.ExtractSingleCells(
        masks=[mask_path],
        image=img_path,
        channel_names=markers,
        output=out,
        n_jobs=2,
    )

    csvs = list(pathlib.Path(out).glob("*.csv"))
    assert len(csvs) == 1, f"Expected 1 CSV, got {csvs}"
    df = pd.read_csv(csvs[0])
    assert len(df) == n_cells, f"Expected {n_cells} rows, got {len(df)}"


def test_centroids_are_global(tmp_path):
    """Centroids must be in global image coordinates, not chunk-local."""
    mask_path, img_path, markers, expected, n_cells = make_synthetic_data(
        tmp_path, img_h=512, img_w=512, tile_size=64  # small tiles → many chunks
    )
    out = str(tmp_path / "output")
    chunkprop.ExtractSingleCells(
        masks=[mask_path], image=img_path, channel_names=markers, output=out, n_jobs=2
    )
    df = pd.read_csv(list(pathlib.Path(out).glob("*.csv"))[0])
    df = df.set_index("CellID")

    for cell_id, (exp_row, exp_col) in expected.items():
        row = df.loc[cell_id]
        assert abs(row["Y_centroid"] - exp_row) < 1.0, (
            f"Cell {cell_id}: Y_centroid {row['Y_centroid']} != expected {exp_row}"
        )
        assert abs(row["X_centroid"] - exp_col) < 1.0, (
            f"Cell {cell_id}: X_centroid {row['X_centroid']} != expected {exp_col}"
        )


def test_column_order(tmp_path):
    """CellID first, morphology tail last, intensity columns in middle."""
    mask_path, img_path, markers, _, _ = make_synthetic_data(tmp_path)
    out = str(tmp_path / "output")
    chunkprop.ExtractSingleCells(
        masks=[mask_path], image=img_path, channel_names=markers, output=out, n_jobs=2
    )
    df = pd.read_csv(list(pathlib.Path(out).glob("*.csv"))[0])

    assert df.columns[0] == "CellID", "CellID must be the first column"

    tail_cols = ("X_centroid", "Y_centroid", "Area", "MajorAxisLength",
                 "MinorAxisLength", "Eccentricity", "Solidity", "Extent", "Orientation")
    present_tail = [c for c in tail_cols if c in df.columns]
    last_cols = list(df.columns[-len(present_tail):])
    assert last_cols == list(present_tail), (
        f"Tail columns out of order.\nExpected: {present_tail}\nGot: {last_cols}"
    )


def test_intensity_values(tmp_path):
    """intensity_mean should match the constant pixel value set per cell per channel."""
    mask_path, img_path, markers, _, n_cells = make_synthetic_data(
        tmp_path, n_channels=2
    )
    out = str(tmp_path / "output")
    chunkprop.ExtractSingleCells(
        masks=[mask_path], image=img_path, channel_names=markers, output=out, n_jobs=2
    )
    df = pd.read_csv(list(pathlib.Path(out).glob("*.csv"))[0]).set_index("CellID")

    for cell_id in range(1, n_cells + 1):
        if cell_id not in df.index:
            continue
        for ch in range(2):
            expected = cell_id * 100 + ch * 10
            got = df.loc[cell_id, f"CH{ch}"]
            assert abs(got - expected) < 1.0, (
                f"Cell {cell_id} CH{ch}: expected {expected}, got {got}"
            )


def test_chunk_size_snap():
    from chunkprop._chunks import compute_chunk_size
    assert compute_chunk_size(1024) == 4096
    assert compute_chunk_size(1240) == 3720
    assert compute_chunk_size(256) == 4096
    assert compute_chunk_size(None) == 4096
    assert compute_chunk_size(None, user_override=2048) == 2048
    try:
        compute_chunk_size(None, user_override=100)  # not divisible by 16
        assert False, "Should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
