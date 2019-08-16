"""
TO DO:
Spectral indices: NDVI,...


"""


import math
import numpy as np
from sklearn import svm


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
        print('Check Input')
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
    LAI = 10.22 * wdvi + 0.4768
    return LAI


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