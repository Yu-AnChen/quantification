"""
chunkprop — chunk-based single-cell spatial quantification.

Public API
----------
Pipeline          : main entry point
ExtractSingleCells / MultiExtractSingleCells : backward-compatible wrappers
"""
import logging

from .pipeline import Pipeline

# Library-level NullHandler: callers configure logging, we don't force any output.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def ExtractSingleCells(
    masks,
    image,
    channel_names,
    output,
    mask_props=None,
    intensity_props=None,
    n_jobs=4,
    chunk_size=None,
    output_format="csv",
):
    """Backward-compatible wrapper matching the original mcquant API."""
    Pipeline(
        mask_paths=list(masks),
        img_path=image,
        marker_csv_path=channel_names,
        output_dir=output,
        mask_props=mask_props,
        intensity_props=intensity_props,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        output_format=output_format,
    ).run()


def MultiExtractSingleCells(
    masks,
    image,
    channel_names,
    output,
    mask_props=None,
    intensity_props=None,
    n_jobs=4,
    chunk_size=None,
    output_format="csv",
):
    """Backward-compatible wrapper (ori used this for the multi-image loop)."""
    logging.getLogger(__name__).info("Extracting single-cell data for %s", image)
    ExtractSingleCells(
        masks, image, channel_names, output,
        mask_props=mask_props,
        intensity_props=intensity_props,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        output_format=output_format,
    )


__all__ = [
    "Pipeline",
    "ExtractSingleCells",
    "MultiExtractSingleCells",
]
