from spacesense.utils import get_NDVI, get_NDWI, get_LAI, get_EVI
import numpy as np
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))
import pytest


def test_get_NDVI():
    """
        Test the get_NDVI function
    """

    """Test shape == 1  : one Pixel"""
    data = np.array([1, 2, 3])
    assert get_NDVI(data, 1, 2) == -1 / 5

    """Test shape == 2  : one line"""

    data = np.ndarray((3, 3))
    data[:, 0] = 1
    data[:, 1] = 2
    data[:, 2] = 3

    assert np.all(get_NDVI(data, 1, 2) == -1 / 5)

    """Test shape == 3  : one image"""

    data = np.ndarray((3, 2, 3))
    data[:, :, 0] = 1
    data[:, :, 1] = 2
    data[:, :, 2] = 3

    ndvi = get_NDVI(data, 1, 2)
    assert ndvi.shape == (3, 2)
    assert np.all(ndvi == -1 / 5)

    """ test edge case : data too large"""
    with pytest.raises(RuntimeError):
        get_NDVI(np.empty((4, 5, 6, 7, 8, 9)))


def test_get_LAI():
    for shape in [ (2, 3), (3, 2, 3)]:
        data = np.empty(shape)
        assert get_LAI(data, 1, 2).shape ==  shape[:-1]

    assert get_LAI(np.array([0.0]*3), 1, 2) == 0.4768

def test_get_NDWI():
    for shape in [(2, 3), (3, 2, 3)]:
        data = np.empty(shape)
        assert get_NDWI(data, 1, 2, 0).shape == shape[:-1]


def test_get_EVI():
    for shape in [(2, 3), (3, 2, 3)]:
        data = np.empty(shape)
        assert get_EVI(data, 1, 2, 0).shape ==  shape[:-1]
