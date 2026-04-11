"""
Image and mask I/O utilities.
"""
import logging
import pathlib
import warnings

import h5py
import pandas as pd
import tifffile

log = logging.getLogger(__name__)


def get_img_metadata(img_path: str) -> dict:
    """
    Return a dict describing the image:
      format     : 'tiff' | 'hdf5'
      n_channels : int
      shape      : (H, W)
      tile_size  : int | None   (None = not tiled or HDF5)
      hdf5_key   : str | None   (HDF5 dataset name, if applicable)
    """
    path = pathlib.Path(img_path)

    if path.suffix in (".h5", ".hdf5"):
        with h5py.File(img_path, "r") as f:
            key = list(f.keys())[0]
            # ori dimension order: (1, H, W, n_channels)
            s = f[key].shape
            return {
                "format": "hdf5",
                "n_channels": s[3],
                "shape": (s[1], s[2]),
                "tile_size": None,
                "hdf5_key": key,
            }

    # TIFF branch
    with tifffile.TiffFile(img_path) as tf:
        series = tf.series[0]
        s = series.shape
        ndim = len(s)
        if ndim == 2:
            n_channels, img_shape = 1, s
        elif ndim == 3:
            n_channels, img_shape = s[0], s[1:]
        else:
            raise ValueError(
                f"Only 2D/3D images are supported; got {ndim}D shape {s}."
            )

        page = series.levels[0].pages[0]
        if page.is_tiled:
            tile_h, tile_w = page.tile[:2]
            if tile_h != tile_w:
                warnings.warn(
                    f"Non-square tiles ({tile_h}×{tile_w}); using the smaller "
                    f"dimension ({min(tile_h, tile_w)}) for chunk alignment.",
                    stacklevel=2,
                )
            tile_size = min(tile_h, tile_w)
        else:
            tile_size = None
            warnings.warn(
                f"{path.name} is not a tiled TIFF. Intensity quantification will "
                "load full channels into RAM. Convert to tiled OME-TIFF for "
                "memory-efficient processing.",
                stacklevel=2,
            )

    return {
        "format": "tiff",
        "n_channels": n_channels,
        "shape": tuple(img_shape),
        "tile_size": tile_size,
        "hdf5_key": None,
    }


def load_marker_csv(csv_path: str) -> list[str]:
    """
    Load channel names from CSV. Supports:
      - Modern format: CSV with a 'marker_name' column.
      - Legacy format: single column, no header.
    Duplicate names are made unique by appending _1, _2, …
    """
    path = pathlib.Path(csv_path)
    df = pd.read_csv(path)
    if "marker_name" not in df.columns:
        log.warning(
            "'marker_name' column not found in %s; assuming legacy single-column format.",
            path.name,
        )
        df = pd.read_csv(path, header=None, usecols=[0], names=["marker_name"])

    dupes = df.duplicated(subset="marker_name", keep=False)
    if dupes.any():
        suffix = (
            df.loc[dupes]
            .groupby("marker_name")
            .cumcount()
            .map(lambda x: f"_{x + 1}")
        )
        df.loc[dupes, "marker_name"] = df.loc[dupes, "marker_name"] + suffix

    return df["marker_name"].tolist()


def validate_masks(mask_paths: list[str]) -> tuple[int, int]:
    """
    Confirm all masks exist, are 2D, and share the same shape.
    Returns (H, W).
    """
    shapes = []
    for p in mask_paths:
        p = pathlib.Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        with tifffile.TiffFile(p) as tf:
            s = tf.series[0].shape
        if len(s) != 2:
            raise ValueError(
                f"Only 2D masks are supported; {p.name} has shape {s}."
            )
        shapes.append(s)

    if len(set(shapes)) != 1:
        detail = "\n".join(f"  {p}: {s}" for p, s in zip(mask_paths, shapes))
        raise ValueError(f"All masks must have the same shape:\n{detail}")

    return shapes[0]


def _stem(path: str) -> str:
    """Return filename stem, stripping .ome and extension (matches ori behaviour)."""
    p = pathlib.Path(path)
    tokens = p.name.split(".")
    if len(tokens) < 2:
        return tokens[0]
    if tokens[-2] == "ome":
        return ".".join(tokens[:-2])
    return ".".join(tokens[:-1])


def write_table(
    df: pd.DataFrame,
    output_dir: str,
    mask_path: str,
    img_path: str,
    output_format: str = "csv",
) -> pathlib.Path:
    """
    Write per-mask quantification table.

    Filename format matches ori: ``{img_stem}_{mask_stem}.csv``
    Written with ``index=False`` (CellID is a regular column).
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_stem = _stem(img_path)
    mask_stem = _stem(mask_path)
    fname = f"{img_stem}_{mask_stem}"

    if output_format == "parquet":
        fpath = out / f"{fname}.parquet"
        df.to_parquet(fpath, index=False)
    else:
        fpath = out / f"{fname}.csv"
        df.to_csv(fpath, index=False)

    return fpath
