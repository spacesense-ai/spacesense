"""
TO DO:
Spectral indices: NDVI,...


"""


import math
import numpy as np


def get_ndvi(data, nir_index=7, red_index=3, datatype='sentinel-2'):
    """
    Normalized difference vegetation index
    NDVI = (b_nir - b_red)/(b_nir + b_red)
    :return: NDVI values for each pixel
    """
    ndvi = (data[:, :, nir_index] - data[:, :, red_index]) / (data[:, :, nir_index] + data[:, :, red_index])

    return ndvi
