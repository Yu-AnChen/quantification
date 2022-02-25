import imagej
import numpy as np
import math

import contextlib
import io

import scyjava
scyjava.config.add_option('-Xmx32g')

def init_ij():
    global ij, BackgroundSubtracter, ImagePlus, bg_subtracter
    if 'ij' not in globals():
        # initialize imagej
        with contextlib.redirect_stdout(io.StringIO()):
            # works on O2
            # ij = imagej.init()
            # FIXME local fiji result in memory leak switching to imagej seems to 
            # resolve the issue
            # ij = imagej.init('/home/yc296/project/imagej-rolling-ball/Fiji.app')
            # works locally
            ij = imagej.init('net.imagej:imageJ:2.1.0')
            # ij = imagej.init('sc.fiji:fiji:2.1.0')
            # ij = imagej.init(r"C:\Users\Public\Downloads\pyimagej\Fiji.app")
        print('ImageJ Version:', ij.getVersion())
    if 'BackgroundSubtracter' in globals(): return
    import jpype
    # get Java class representations
    BackgroundSubtracter = jpype.JClass(
        'ij.plugin.filter.BackgroundSubtracter'
    )
    ImagePlus = jpype.JClass('ij.ImagePlus')
    bg_subtracter = BackgroundSubtracter()
    return


def ij_rolling_ball(img, radius, verbose=None):
    init_ij()

    src_dtype = img.dtype
    src_shape = img.shape

    imp = ij.py.to_java(img)
    imp = ij.dataset().create(imp)
    imp = ij.convert().convert(imp, ImagePlus)

    del img

    bg_subtracter.rollingBallBackground(
        imp.getProcessor(), radius, 
        False, False, False, False, True
    )

    subtracted = np.array(
        imp.getProcessor().getPixels(),
        dtype=src_dtype
    ).reshape(src_shape)

    imp.close()
    return subtracted


def compute_chunk_size_and_overlap(chunk_size, overlap_depth, img_size):
    # small image
    if (img_size <= chunk_size) or (img_size <= overlap_depth):
        return img_size, 0
    # small chunk
    if chunk_size <= overlap_depth:
        chunk_size = overlap_depth
    chunk_size = math.ceil(chunk_size / 8) * 8
    if img_size % chunk_size < overlap_depth:
        # increase chunk_size to have more "space" for the remainder
        chunk_size += 8
        return compute_chunk_size_and_overlap(chunk_size, overlap_depth, img_size)
    else:
        return chunk_size, overlap_depth


def ij_rolling_ball_dask(img, radius, chunk_size=2**11, verbose=True):
    import dask.array as da
    import dask.diagnostics

    # find the right chunk sizes to generate consistent result
    # FIXME one may want to "lock" chunk sizes in some use cases
    h, w = img.shape
    overlap_depth = radius * get_shrink_factor(radius)    
    chunk_h, overlap_h = compute_chunk_size_and_overlap(chunk_size, overlap_depth, h)
    chunk_w, overlap_w = compute_chunk_size_and_overlap(chunk_size, overlap_depth, w)
    
    chunk_shape = (chunk_h, chunk_w)
    depth = {0: overlap_h, 1: overlap_w}

    if verbose:
        print(f"Image shape: {img.shape}")
        if (chunk_size != chunk_h) or (chunk_size != chunk_w):
            print(
                f"Adjusted chunk shape {chunk_shape}; was ({chunk_size}, {chunk_size})\n"
                f"Number of chunks: "
                f"{tuple(np.ceil(np.divide(img.shape, chunk_shape)).astype(int))}"
            )

    da_img = da.from_array(img, chunks=chunk_shape)
    da_result = da.map_overlap(
        ij_rolling_ball, da_img, dtype=img.dtype, 
        depth=depth, boundary='none', radius=radius,
    )

    if verbose:
        with dask.diagnostics.ProgressBar():
            return da_result.compute()
    else:
        return da_result.compute()


def get_random_img(shape, dtype=np.uint16):
    dtype = np.dtype(dtype)
    assert np.issubdtype(dtype, np.integer)
    random_img = np.random.randint(
        np.iinfo(dtype).min, np.iinfo(dtype).max, size=shape,
        dtype=dtype
    )
    return random_img


# https://github.com/imagej/imagej1/blob/master/ij/plugin/filter/BackgroundSubtracter.java#L779-L795
def get_shrink_factor(radius):
    if radius <= 10:
        shrinkFactor = 2
    elif radius <= 30:
        shrinkFactor = 2
    elif radius <= 100:
        shrinkFactor = 4
    else:
        shrinkFactor = 8
    return shrinkFactor


def test(
    shape=None,
    radius=None,
    chunk_size=None,
):
    import matplotlib.pyplot as plt
    if shape is None:
        shape = np.random.randint(1000, 2000, size=2)
    if radius is None:
        radius = np.random.randint(1, 150)
    if chunk_size is None:  
        chunk_size = round(min(shape) / 2)
    test_img = get_random_img(shape=shape, dtype=np.uint16)
    print(
        f"shape={test_img.shape}, radius={radius}, chunk_size={chunk_size}"
    )
    raw = ij_rolling_ball(test_img, radius=radius)
    chunked = ij_rolling_ball_dask(
        test_img, radius=radius, 
        chunk_size=chunk_size
    )
    print('Number of different pixels:', np.sum(raw != chunked))
    plt.figure()
    plt.imshow(raw != chunked)
    plt.suptitle(f'{test_img.shape}; r = {radius}; s = {chunk_size}')