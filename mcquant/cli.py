"""
Command-line interface for mcquant.
"""
import argparse
import sys

from .pipeline import Pipeline


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="mcquant",
        description="Single-cell spatial quantification from multiplexed imaging.",
    )

    # Required args (matches ori interface)
    parser.add_argument(
        "--mask", "-m",
        dest="masks",
        nargs="+",
        required=True,
        metavar="MASK",
        help="One or more segmentation mask TIFF paths.",
    )
    parser.add_argument(
        "--image", "-i",
        dest="image",
        required=True,
        metavar="IMAGE",
        help="Multichannel OME-TIFF or HDF5 image path.",
    )
    parser.add_argument(
        "--channel-names", "-c",
        dest="channel_names",
        required=True,
        metavar="CSV",
        help="CSV file with channel names (marker_name column or legacy format).",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output",
        required=True,
        metavar="DIR",
        help="Output directory.",
    )

    # Optional processing args
    parser.add_argument(
        "--mask-props",
        dest="mask_props",
        nargs="+",
        default=None,
        metavar="PROP",
        help="Additional morphological properties (skimage regionprops names).",
    )
    parser.add_argument(
        "--intensity-props",
        dest="intensity_props",
        nargs="+",
        default=None,
        metavar="PROP",
        help=(
            "Extra intensity properties beyond intensity_mean. "
            "Available: gini_index, intensity_median, intensity_sum."
        ),
    )

    # Parallelism / chunking
    parser.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=None,
        metavar="PIXELS",
        help=(
            "Chunk size in pixels (must be divisible by 16). "
            "Defaults to nearest multiple of the image tile size to ~4096."
        ),
    )

    # Output format
    parser.add_argument(
        "--output-format",
        dest="output_format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format (default: csv).",
    )

    args = parser.parse_args(argv)

    pipeline = Pipeline(
        mask_paths=args.masks,
        img_path=args.image,
        marker_csv_path=args.channel_names,
        output_dir=args.output,
        mask_props=args.mask_props,
        intensity_props=args.intensity_props,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        output_format=args.output_format,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
