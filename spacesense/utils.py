
import math
import numpy as np
from sklearn import svm
import os

def get_NDVI(data, nir_index=7, red_index=3, datatype='sentinel-2'):
    """
    Normalized difference vegetation index
    NDVI = (b_nir - b_red)/(b_nir + b_red)
    :return: NDVI values for each pixel
    """
    if len(data.shape)==3:
        ndvi = (data[:, :, nir_index] - data[:, :, red_index]) / (data[:, :, nir_index] + data[:, :, red_index])
    elif len(data.shape)==2:
        ndvi = (data[:, nir_index] - data[:, red_index]) / (data[:, nir_index] + data[:, red_index])
    elif len(data.shape)==1:
        ndvi = (data[nir_index] - data[red_index]) / (data[nir_index] + data[red_index])
    else:
        raise RuntimeError('Check Input')
    return ndvi

def get_LAI(data, nir_index=7, red_index=3, datatype='sentinel-2'):
    """
    Leaf Area Index
        source  : Clevers, J.G.P.W.; Kooistra, L.; van den Brande, M.M.M. ,
        Using Sentinel-2 Data for Retrieving LAI and Leaf and Canopy Chlorophyll Content of a Potato Crop. ,
        Remote Sens. 2017, 9, 405.
    :return:
    """
    Cwdvi = 1.35
    if len(data.shape) == 3:
        B08 = data[:,:,nir_index]
        B04 = data[:,:,red_index]
    elif len(data.shape)==2:
        B08 = data[:,nir_index]
        B04 = data[:,red_index]
    elif len(data.shape)==1:
        B08 = data[nir_index]
        B04 = data[red_index]
    else:
        print('Check Input')
    wdvi = B08 - Cwdvi * B04
    lai = 10.22 * wdvi + 0.4768
    return lai

def get_NDWI(data, nir_index=7, green_index=2,swir_index=11, option=1):
    """
    Normalized difference water index

    option 1:
    NDWI = (b_green - b_nir)/(b_nir + b_green)

    option 2:
    NDWI = (b_nir - b11_swir)/(b_nir + b11_swir)
    :return: NDWI values for each pixel
    """
    if len(data.shape) == 3:
        B08 = data[:, :, nir_index]
        B03 = data[:, :, green_index]
        B11 = data[:, :, swir_index]
    elif len(data.shape)==2:
        B08 = data[:, nir_index]
        B03 = data[:, green_index]
        B11 = data[:, swir_index]
    elif len(data.shape)==1:
        B08 = data[nir_index]
        B03 = data[green_index]
        B11 = data[swir_index]
    else:
        print('Check input')
    if option==1:
        ndwi = (B03 - B08) / (B08 + B03)

    elif option==2:
        """
        needs resampling band 11 SWIR to 10x10 or band 8 NIR to 20x20
        """
        ndwi = (B08 - B11) / (B08 + B11)

    return ndwi


def get_EVI(data, nir_index=7, red_index=3,blue_index=1 ,datatype='sentinel-2'):
    """
    Enhanced Vegetation Index
        source  : https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=16
    :return: EVI
    """
    if len(data.shape) == 3:
        B08 = data[:,:,nir_index]
        B04 = data[:,:,red_index]
        B02 = data[:, :,blue_index]
    elif len(data.shape)==2:
        B08 = data[:,nir_index]
        B04 = data[:,red_index]
        B02 = data[:,blue_index]
    elif len(data.shape)==1:
        B08 = data[nir_index]
        B04 = data[red_index]
        B02 = data[blue_index]
    else:
        print('Check Input')
    evi = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
    return evi

def optimize_OneClassSVM(X, n):
    print('searching for optimal hyperparameters...')
    nu = np.linspace(start=1e-5, stop=1e-2, num=n)
    gamma = np.linspace(start=1e-6, stop=1e-3, num=n)
    opt_diff = 1.0
    opt_nu = None
    opt_gamma = None
    for i in range(len(nu)):
        for j in range(len(gamma)):
            classifier = svm.OneClassSVM(kernel="rbf", nu=nu[i], gamma=gamma[j])
            classifier.fit(X)
            label = classifier.predict(X)
            p = 1 - float(sum(label == 1.0)) / len(label)
            diff = math.fabs(p - nu[i])
            if diff < opt_diff:
                opt_diff = diff
                opt_nu = nu[i]
                opt_gamma = gamma[j]
    return opt_nu, opt_gamma

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)