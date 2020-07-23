"""
Purpose: Tests for "geoimage.py"
Author:  Ken Youens-Clark <kyclark@arizona.edu>
"""

import os
import numpy as np
import random
import string
import pytest
import tempfile
from .geoimage import clip_raster

input_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'test_data', 'canopy'))
orthomosaic = os.path.join(input_dir, 'orthomosaic.tif')
bounds = (3660122.578269566, 3660126.3783758446, 408988.9185874542,
          408990.26284273656)


# --------------------------------------------------
def test_input_dir() -> None:
    """Good inputs"""

    assert os.path.isdir(input_dir)


# --------------------------------------------------
def test_clip_raster_ok1() -> None:
    """Good input"""

    assert os.path.isfile(orthomosaic)
    px = clip_raster(orthomosaic, bounds)
    assert isinstance(px, np.ndarray)


# --------------------------------------------------
def test_clip_raster_bad_file() -> None:
    """Non-existent file"""

    bad = random_string()

    with pytest.raises(Exception, match=f'Bad input raster path "{bad}"'):
        px = clip_raster(bad, bounds)


# --------------------------------------------------
def test_clip_raster_empty_file() -> None:
    """Empty file"""

    empty = tempfile.NamedTemporaryFile(delete=False, mode='wt', suffix='.tif')
    empty.close()

    print(empty.name)
    with pytest.raises(Exception, match='foo'):
        px = clip_raster(empty.name, bounds)

    os.remove(empty.name)


# --------------------------------------------------
def _test_clip_raster() -> None:
    """Test clip_raster"""

    bad_file = random_string()
    bad_bounds = ('foo', 'bar')
    assert clip_raster(bad_file, bad_bounds) == None


# --------------------------------------------------
def random_string():
    """generate a random string"""

    k = random.randint(5, 10)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))
