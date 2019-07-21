"""
CLASSIFICATION MODULE

"""
import os
import sentinelsat as ss
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import matplotlib.pyplot as plt
from datetime import date
from osgeo import gdal
import numpy as np
from glob import glob
import time


class download_sentinel(object):
    def __init__(self, username, password):

        self.username = username
        self.password = password
        self.roi_polygon = None
        self.startdate = None
        self.enddate = None
        self.list_products = None

    def sentinel(self, download_type='ROI_polygon', roi_polygon=None, startdate=None, enddate=None, cloudcover_max=5,
                 platformname='Sentinel-2'):
        '''

        :param download_type:
        :param username:
        :param password:
        :param roi_polygon:
        :param startdate:
        :param enddate:
        :param cloudcover_max:
        :param platformname:
        :return:
        '''

        if startdate:
            self.startdate = startdate
        if enddate:
            self.enddate = enddate

        if roi_polygon:
            self.roi_polygon = roi_polygon
        self.api = SentinelAPI(self.username, self.password, 'https://scihub.copernicus.eu/dhus')

        product_id = None
        if download_type == 'full':
            if product_id:
                self.api.download(product_id)
            else:
                print('product id required')

        if download_type == 'ROI_polygon':
            if roi_polygon.split('.')[-1] == 'geojson':
                footprint = geojson_to_wkt(read_geojson(self.roi_polygon))

                products = self.api.query(footprint,
                                          date=(self.startdate, self.enddate),
                                          platformname='Sentinel-2',
                                          cloudcoverpercentage=(0, cloudcover_max))
                self.list_products = list(products.items())
        print(len(self.list_products), ' products found')

    def download_files(self, list_product_ids,directory_path='.'):
        for product_id in list_product_ids:
            self.api.download(product_id, directory_path=directory_path)

    """ add function to display product AOI Polygon"""


class download_modis(object):
    """

    TBD
    :return:
    """

    def __init__(self):
        """
        """




class read_modis(object):
    def __init__(self, hdf_data_path):
        self.hdf_data_path = hdf_data_path
        df = gdal.Open(self.hdf_data_path, gdal.GA_ReadOnly)
        self.meta_data = df.GetMetadata()
        self.sdf = df.GetSubDatasets()
        self.band_details = [self.sdf[i][1] for i in range(len(self.sdf))]
        self.num_bands = len(self.sdf)
        self.img_shp = gdal.Open(self.sdf[0][0]).ReadAsArray().shape
        self.AOI = None
        self.data = None

    def get_data(self):
        self.data = np.zeros((self.img_shp[0], self.img_shp[1], self.num_bands))
        for i in range(self.num_bands):
            self.data[:, :, i] = gdal.Open(self.sdf[i][0]).ReadAsArray()

    def save_as_npy(self, save_folder=os.getcwd()):
        np.save(save_folder + '/data', self.data)
        print('dataset saved in .npy format at this location:', save_folder)


class read_sentinel(object):

    def __init__(self, folder_path, img_type='sentinel_2'):
        self.img_type = img_type
        self.folder_path = folder_path
        self.band_files = sorted(glob(self.folder_path + '/*.jp2'))
        self.band_details = [file.split('_')[-1].split('.')[0] for file in self.band_files]
        im = gdal.Open(self.band_files[1])
        arf_base = im.ReadAsArray()
        self.img_shp = arf_base.shape
        self.meta_data = im.GetMetadata()
        self.num_bands = len(self.band_files)
        self.AOI = None
        self.data = None

    def get_data(self, bandfiles, row_start=0, col_start=0, row_len=1000, col_len=1000):
        """
        1. loads from npy files for efficiency
        2. order of bands: [01,02,03,04,05,06,07,08,09,10,11,12,8A]

        """
        if bandfiles == None:
            bandfiles = self.band_files
        npy_files = sorted(glob(self.folder_path + '/*.npy'))
        if len(npy_files) == 0:
            if self.img_type == 'sentinel-2':
                self.sentinel_2_remap()

        self.row_len = row_len
        self.col_len = col_len
        self.row_start = row_start
        self.col_start = col_start
        input_bands = bandfiles
        num_bands = len(input_bands)
        data = np.zeros((self.row_len, self.col_len, num_bands))
        # one band info for all pixels loaded in each iteration
        for i in range(num_bands):
            arf = np.load(input_bands[i])
            ap1 = arf[row_start:row_start + row_len, col_start:col_start + col_len]
            data[:, :, i] = ap1

    def save_as_npy(self, save_folder=os.getcwd()):
        np.save(save_folder + '/data', self.data)
        print('dataset saved in .npy format at this location:', save_folder)

    def sentinel_2_remap(self):
        # load and save all bands as numpy arrays for resuse and remap bands [3,4,5,7,8,9]
        start_time = time.time()
        row_, col_ = self.img_shp
        for b in range(10):
            if b in [4, 5, 6, 10, 11, 12]:
                ar_n = np.zeros((row_, col_))
                im = gdal.Open(self.band_files[b])
                ar = im.ReadAsArray()
                name = self.band_details[b] + '.npy'
                for i in range(int(row_ / 2)):
                    for j in range(int(col_ / 2)):
                        n = int(i * 2)
                        m = int(j * 2)
                        ar_n[n:n + 2, m:m + 2] = ar[i, j]
                np.save(self.folder_path + '/' + name, ar_n)
                print(self.band_details[b], ar_n.shape)

            else:
                im = gdal.Open(self.band_files[b])
                ar = im.ReadAsArray()
                name = self.band_details[b] + '.npy'
                np.save(self.folder_path + '/' + name, ar)
                print(self.band_details[b], ar.shape)
        print('all bands saved as numpy arrays for faster processing in \
              %s seconds' % (time.time() - start_time))
    @staticmethod
    def get_ndvi(data):
        """
        NDVI = (b_nir - b_red)/(b_nir + b_red)
        :return: NDVI values for each pixel
        """
        ndvi = (data[:,:,7] - data[:,:,3])/(data[:,:,7] + data[:,:,3])

        return ndvi


