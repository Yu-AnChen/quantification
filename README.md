# mcquant

Single-cell spatial quantification from multiplexed imaging.  
Extracts per-cell morphological and intensity measurements from segmentation masks and multi-channel images. Output is compatible with histoCAT.

## How it works

Large images are processed in overlapping spatial chunks rather than loaded into memory all at once. The required overlap between chunks is computed automatically from the mask so that no cell is split across chunk boundaries. RAM usage scales with the number of parallel workers and chunk size, not with image size.

```
RAM ≈ n_jobs × chunk_size² × bytes_per_pixel
```

Images must be **tiled OME-TIFF** for full benefit. Non-tiled TIFFs and HDF5 files are supported with a fallback to full-channel reads.

## Installation

```bash
pip install .
```

Or with [pixi](https://pixi.sh):

```bash
pixi install
```

## CLI usage

```bash
mcquant \
  --mask  segmentation/cellRing.ome.tif \
  --image registration/image.ome.tif \
  --channel-names markers.csv \
  --output feature_extraction/
```

Multiple masks are each quantified independently:

```bash
mcquant \
  --mask segmentation/cellRing.ome.tif segmentation/nucleiMask.ome.tif \
  --image registration/image.ome.tif \
  --channel-names markers.csv \
  --output feature_extraction/
```

### All options

| Option | Default | Description |
|---|---|---|
| `--mask` | required | One or more segmentation mask TIFF paths (int32) |
| `--image` | required | Multichannel OME-TIFF or HDF5 image |
| `--channel-names` | required | CSV with `marker_name` column (or legacy single-column format) |
| `--output` | required | Output directory |
| `--n-jobs` | `4` | Number of parallel workers |
| `--chunk-size` | auto | Chunk size in pixels — must be divisible by 16. Defaults to the nearest multiple of the image tile size that is close to 4096 (e.g. tile=1240 → chunk=3720) |
| `--mask-props` | — | Extra morphological properties (skimage `regionprops` names) |
| `--intensity-props` | — | Extra intensity properties (see below) |
| `--output-format` | `csv` | `csv` or `parquet` |

### Channel names CSV

Modern format (recommended):

```csv
marker_name
DAPI
CD3
CD8
...
```

Legacy single-column format (no header) is also accepted.  
Duplicate names are automatically suffixed: `CD8`, `CD8_1`, `CD8_2`, …

## Output

One file per mask, named `{image}_{mask}.csv` (or `.parquet`), written to the output directory.

Column layout matches histoCAT convention:

```
CellID | <channel columns> | X_centroid | Y_centroid | Area | MajorAxisLength | ...
```

- **CellID** — integer cell label from the mask
- **Channel columns** — `intensity_mean` stored under the bare channel name (`CD8`, `DAPI`); extra properties suffixed (`CD8_gini_index`)
- **Morphology tail** — spatial and shape properties last

## Extra intensity properties

Specify with `--intensity-props`:

| Name | Description |
|---|---|
| `gini_index` | Gini coefficient of pixel intensities within the cell (0 = uniform, 1 = maximally unequal) |
| `intensity_median` | Median pixel intensity |
| `intensity_sum` | Sum of pixel intensities — useful for counting RNA molecules from FISH images |

Example:

```bash
mcquant --mask ... --image ... --channel-names ... --output ... \
        --intensity-props gini_index intensity_median
```

## Python API

```python
from mcquant import Pipeline

Pipeline(
    mask_paths=["segmentation/cellRing.ome.tif"],
    img_path="registration/image.ome.tif",
    marker_csv_path="markers.csv",
    output_dir="feature_extraction/",
    n_jobs=8,
    intensity_props=["gini_index"],
).run()
```

Backward-compatible wrappers matching the original mcquant API are also available:

```python
from mcquant import MultiExtractSingleCells

MultiExtractSingleCells(
    masks=["segmentation/cellRing.ome.tif"],
    image="registration/image.ome.tif",
    channel_names="markers.csv",
    output="feature_extraction/",
)
```

## Image format notes

| Format | Chunk reads | Recommended |
|---|---|---|
| Tiled OME-TIFF | Yes — subregion reads via zarr | ✓ |
| Non-tiled TIFF | No — full channel loaded per worker | convert with `bioformats2raw` or `tifffile` |
| HDF5 | Yes — h5py slice reads | ✓ |

To convert a non-tiled TIFF to tiled OME-TIFF with Python:

```python
import tifffile, numpy as np
img = tifffile.imread("image.tif")
tifffile.imwrite("image.ome.tif", img, tile=(1024, 1024), compression="zlib")
```
