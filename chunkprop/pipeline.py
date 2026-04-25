"""
Pipeline: orchestrates mask loading, zarr conversion, overlap computation,
morphology pass, per-channel intensity passes, and CSV/parquet output.
"""
import logging
import os
import tempfile

import numpy as np

from ._chunks import compute_chunk_size, compute_overlap, mask_to_zarr
from ._io import get_img_metadata, load_marker_csv, validate_masks, write_table
from ._props import apply_precision, validate_props
from ._quantify import build_output, intensity_pass, morphology_pass

log = logging.getLogger(__name__)


class Pipeline:
    """
    Parameters
    ----------
    mask_paths : list of str
        Paths to segmentation mask TIFFs (int32, one per mask type).
    img_path : str
        Path to multichannel OME-TIFF or HDF5 image.
    marker_csv_path : str
        CSV with channel names (``marker_name`` column or legacy single-column).
    output_dir : str
        Directory for output files.
    mask_props : list of str, optional
        Extra morphological properties to compute (skimage names).
    intensity_props : list of str, optional
        Extra intensity properties beyond ``intensity_mean``
        (e.g. ``gini_index``, ``intensity_median``, ``intensity_sum``).
    n_jobs : int
        Number of parallel workers.
    chunk_size : int, optional
        Override the automatic chunk size (must be divisible by 16).
    output_format : {'csv', 'parquet'}
        Output file format.
    """

    def __init__(
        self,
        mask_paths: list[str],
        img_path: str,
        marker_csv_path: str,
        output_dir: str,
        mask_props: list[str] | None = None,
        intensity_props: list[str] | None = None,
        n_jobs: int = 4,
        chunk_size: int | None = None,
        output_format: str = "csv",
    ) -> None:
        validate_props(mask_props)
        validate_props(intensity_props)
        mask_shape = validate_masks(mask_paths)

        self.mask_paths = mask_paths
        self.img_path = img_path
        self.marker_csv_path = marker_csv_path
        self.output_dir = output_dir
        self.mask_props = mask_props
        self.intensity_props = intensity_props
        self.n_jobs = n_jobs
        self.output_format = output_format

        img_meta = get_img_metadata(img_path)
        channel_names = load_marker_csv(marker_csv_path)

        n_img_channels = img_meta["n_channels"]
        n_names = len(channel_names)
        if n_names != n_img_channels:
            raise ValueError(
                f"Channel name count ({n_names}) does not match "
                f"image channel count ({n_img_channels}) in {img_path}."
            )
        if tuple(img_meta["shape"]) != tuple(mask_shape):
            raise ValueError(
                f"Image spatial shape {img_meta['shape']} does not match "
                f"mask shape {mask_shape}."
            )

        self._img_meta = img_meta
        self._channel_names = channel_names
        self._mask_shape = mask_shape
        self._chunk_size = compute_chunk_size(img_meta["tile_size"], chunk_size)

        # Determine image read mode for intensity workers
        if img_meta["format"] == "hdf5":
            self._img_mode = "hdf5"
        elif img_meta["tile_size"] is not None:
            self._img_mode = "tiff_tiled"
        else:
            self._img_mode = "tiff_nontiled"

        log.info(
            "Image: %d channels, shape %s, tile_size=%s, chunk_size=%d, mode=%s",
            img_meta["n_channels"],
            img_meta["shape"],
            img_meta["tile_size"],
            self._chunk_size,
            self._img_mode,
        )

    def run(self) -> None:
        for mask_path in self.mask_paths:
            log.info("Processing mask: %s", os.path.basename(mask_path))
            self._run_mask(mask_path)

    def _run_mask(self, mask_path: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_zarr_dir = os.path.join(tmp_dir, "mask.zarr")

            log.info("Converting mask to zarr store")
            mask_to_zarr(mask_path, self._chunk_size, mask_zarr_dir)

            # Compute required overlap
            overlap = compute_overlap(
                mask_zarr_dir, self._mask_shape, self._chunk_size, self.n_jobs
            )
            log.info("Overlap: %d px", overlap)

            # Morphology pass
            morph_df = morphology_pass(
                mask_zarr_dir,
                self._mask_shape,
                self._chunk_size,
                overlap,
                self.mask_props,
                self.n_jobs,
            )
            valid_labels = morph_df["CellID"].to_numpy(dtype=np.int64)
            max_label = int(valid_labels.max())
            log.info("Cells found: %d  (max label: %d)", len(valid_labels), max_label)

            # Intensity passes — one per channel
            n_channels = len(self._channel_names)
            channel_arrays: dict[str, np.ndarray] = {}
            for ch_idx, ch_name in enumerate(self._channel_names):
                log.info("Channel %d/%d: %s", ch_idx + 1, n_channels, ch_name)
                ch_arrays = intensity_pass(
                    mask_zarr_dir=mask_zarr_dir,
                    img_path=self.img_path,
                    img_format=self._img_mode,
                    hdf5_key=self._img_meta["hdf5_key"],
                    shape=self._mask_shape,
                    channel_idx=ch_idx,
                    channel_name=ch_name,
                    chunk_size=self._chunk_size,
                    overlap=overlap,
                    valid_labels=valid_labels,
                    max_label=max_label,
                    intensity_props=self.intensity_props,
                    n_jobs=self.n_jobs,
                )
                channel_arrays.update(ch_arrays)

        # Build final DataFrame (one allocation, outside tempdir)
        result = build_output(morph_df, channel_arrays)
        result = apply_precision(result)

        out_path = write_table(
            result,
            self.output_dir,
            mask_path,
            self.img_path,
            self.output_format,
        )
        log.info("Written: %s", out_path)
