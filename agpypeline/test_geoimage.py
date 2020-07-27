"""
Purpose: Tests for "geoimage.py"
Author:  Ken Youens-Clark <kyclark@arizona.edu>
"""

import os
import random
import string
import tempfile
import numpy as np
import pytest
from .geoimage import clip_raster, clip_raster_intersection

input_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'test_data', 'canopy'))
orthomosaic = os.path.join(input_dir, 'orthomosaic.tif')
bounds = (3660122.578269566, 3660126.3783758446, 408988.9185874542,
          408990.26284273656)


# --------------------------------------------------
def test_input_dir() -> None:
    """Good inputs"""

    assert os.path.isdir(input_dir)


# --------------------------------------------------
def test_clip_raster_ok_no_out_file() -> None:
    """
    Test with a known, good input file.
    """

    assert os.path.isfile(orthomosaic)
    pixels = clip_raster(orthomosaic, bounds)
    assert isinstance(pixels, np.ndarray)


# --------------------------------------------------
def test_clip_raster_out_file() -> None:
    """
    Test with a known, good input file.
    """

    out_fh = tempfile.NamedTemporaryFile(delete=True, mode='wt')
    out_fh.close()

    try:
        pixels = clip_raster(raster_path=orthomosaic,
                             bounds=bounds,
                             out_path=out_fh.name)
        assert isinstance(pixels, np.ndarray)
        assert os.path.isfile(out_fh.name)

    finally:
        if os.path.isfile(out_fh.name):
            os.remove(out_fh.name)


# --------------------------------------------------
def test_clip_raster_bad_file() -> None:
    """
    Test with a non-existent file.
    The code never actually checks if the given file exists, so
    the error generated doesn't reflect the actual problem.
    Instead, the function `gdal_translate` call fails to produce
    an output file, and then the Python generates an exception when it
    tries to read an empty/nonexistent output file.
    """

    bad = random_string()
    err = "'NoneType' object has no attribute 'ReadAsArray'"

    with pytest.raises(Exception, match=err):
        _ = clip_raster(bad, bounds)


# --------------------------------------------------
def test_clip_raster_too_few_bounds() -> None:
    """
    The function doesn't bother to check the size bounds is equal to 4,
    so this throws an exception.
    """

    for n in range(1, 4):
        bad_bounds = [random.random() for _ in range(n)]

        with pytest.raises(IndexError, match='list index out of range'):
            assert clip_raster(orthomosaic, bad_bounds) is None


# --------------------------------------------------
def test_clip_raster_intersection() -> None:
    """
    (file_path: str,
     file_bounds: ogr.Geometry,
     plot_bounds: ogr.Geometry,
     out_file: str)
    """

    assert clip_raster_intersection(orthomosaic, bounds, bounds, 'foo')


# --------------------------------------------------
def random_string():
    """generate a random string"""

    k = random.randint(5, 10)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))
