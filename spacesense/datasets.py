"""
DATASETS: download and perform basic operations

"""
import os
import time
from glob import glob
import numpy as np
from osgeo import gdal
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from rasterio.enums import Resampling
import rasterio
from .utils import clip_to_aoi
import zipfile
import json
import codecs



class Dataset_general():
    """General class to define the needed functions to define in a Dataset class

    The Dataset Class are used to fetch and download images"""

    def __init__(self):
        """init the Dataset class. May be use to set credentials for the API"""

        self.roi_polygon = None
        self.startdate = None
        self.enddate = None
        self.list_products = None


    def fetch_datasets(self, download_type='ROI_polygon', roi_polygon=None, startdate=None, enddate=None, cloudcover_max=5):
        """Make a request to the API to obtain the availables images

        :param download_type: str to specify the product fetched. Can be "ROI_polygon" or "full"
        :param roi_polygon: str : the filename to the ROI_polygon. Needs to by a geojson format
        :param startdate: datetime.date object for the start of the lookup
        :param enddate: datetime.date object for the endof the lookup
        :param cloudcover_max: The maximum cloud coverage allowed

        Sets self.list_products to the list of the product ID  for the API to download
        """

        raise RuntimeError("method not implemented")

    def download(self, list_product_ids,directory_path='.'):
        """

        :param list_product_ids: list of the product_ID to download
        :param directory_path: str path location
        """
        raise RuntimeError("method not implemented")


class download_sentinel(Dataset_general):
    def __init__(self, username, password):

        super().__init__()
        self.username = username
        self.password = password

    def sentinel_2(self,download_type='ROI_polygon', roi_polygon=None, startdate=None, enddate=None, cloudcover_max=5):
        params = {'download_type':download_type,'roi_polygon': roi_polygon, 'startdate': startdate, 'enddate': enddate,'platformname': 'Sentinel-2'}
        return self.fetch_datasets(**params)

    def sentinel_1(self,download_type='ROI_polygon', roi_polygon=None, startdate=None, enddate=None):
        params = {'download_type':download_type,'roi_polygon': roi_polygon, 'startdate': startdate, 'enddate': enddate,'platformname': 'Sentinel-1'}
        return self.fetch_datasets(**params)

    def fetch_datasets(self, download_type='ROI_polygon', roi_polygon=None, startdate=None, enddate=None, cloudcover_max=5,
                       platformname='Sentinel-2'):
        """

        :param download_type:
        :param username:
        :param password:
        :param roi_polygon:
        :param startdate:
        :param enddate:
        :param cloudcover_max:
        :param platformname:
        :return:
        """

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
                
                file_obj = open(self.roi_polygon, "r")
                json_data = file_obj.read()
                file_obj.close()
                json_data = json_data.encode().decode('utf-8-sig')  # Remove utf-8 data if any present in the file
                json_data = json.loads(json_data)
                footprint = geojson_to_wkt(json_data)

                if platformname == 'Sentinel-2':
                    self.products = self.api.query(footprint,
                                                date=(self.startdate, self.enddate),
                                                platformname=platformname,
                                                cloudcoverpercentage=(0, cloudcover_max))
                    self.list_products = list(self.products.items())

                elif platformname == 'Sentinel-1':
                    self.products = self.api.query(footprint,
                                                date=(self.startdate, self.enddate),
                                                platformname=platformname)
                    self.list_products = list(self.products.items())
                    
        print(len(self.list_products), ' products found')

    def download_files(self, list_product_ids,directory_path='.',unzip=True):
        for product_id in list_product_ids:
            self.api.download(product_id, directory_path=directory_path)

        if unzip:
            print('extracting files')
            file_names =glob(os.path.join(directory_path)+'/S*.zip')
            for filename in file_names:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(directory_path)

    """ add function to display product AOI Polygon"""


class download_modis(Dataset_general):
    """
    The MODIS instrument is operating on both the Terra and Aqua spacecraft.
    It has a viewing swath width of 2,330 km and views the entire surface of the Earth every one to two days.
    Its detectors measure 36 spectral bands between 0.405 and 14.385 µm, and it acquires data at three spatial resolutions -- 250m, 500m, and 1,000m.

    :return:
    """

    def __init__(self, username, password):
        super(download_modis, self).__init__()

        self.username = username
        self.password = password
        self.product_shortname = None

    def fetch_datasets(self, download_type='ROI_polygon',
                       roi_polygon=None,
                       startdate=None,
                       enddate=None,
                       cloudcover_max=5,
                       product_shortname="MOD15A2H",
                       max_products=-1,
                       inverse_polygon_order=False):

        """Qary the NASA API for products

        See: https://modis.ornl.gov/data/modis_webservice.html

        """
        from cmr import GranuleQuery

        self.product_shortname = product_shortname
        self.roi_polygon = roi_polygon

        if download_type == 'ROI_polygon':
            if roi_polygon.split('.')[-1] == 'geojson':
                list_coords = from_geojson_to_list_coords(self.roi_polygon)
                #print(list_coords)

        else:
            raise RuntimeError("Unknown download type")

        if inverse_polygon_order:
            list_coords = list_coords[::-1]

        api = GranuleQuery().polygon(list_coords).short_name(self.product_shortname).temporal(startdate, enddate)

        n_produtes = api.hits()
        #print(f"{n_produtes} products found for these parameters")

        if max_products == -1:
            max_products = n_produtes

        self.list_products = api.get(limit=max_products)
        self.list_products_id = [ f["producer_granule_id"] for f in self.list_products]

    def download_files(self, list_product_ids,
                       directory_path='.',
                       verbose=True):
        """download the products"""

        import requests
        import sys
        import logging
        import wget

        LOG = logging.getLogger(__name__)
        OUT_HDLR = logging.StreamHandler(sys.stdout)
        OUT_HDLR.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        OUT_HDLR.setLevel(logging.INFO)
        LOG.addHandler(OUT_HDLR)
        LOG.setLevel(logging.INFO)


        CHUNKS = 65536

        with requests.Session() as s:
            s.auth = (self.username, self.password)

            for product_id in list_product_ids:

                if verbose:
                    print("Downloading prodcut:", product_id)
                index_product = self.list_products_id.index(product_id)

                product_name = ".".join(product_id.split(".")[1:-1])

                """Create the folder for the product"""
                os.mkdir(os.path.join(directory_path, product_name), )

                product_links = self.list_products[index_product]["links"]

                url_jpgs = []
                url_file = None
                for p_link in product_links:
                    if "type" in p_link:
                        if p_link["type"] == 'application/x-hdfeos':
                            url_file = p_link["href"]
                        elif p_link["type"] == "image/jpeg":
                            if verbose:
                                print("The link object is", p_link)
                            url_jpgs.append(p_link["href"])


                file_name = os.path.join(directory_path, product_name, product_id, )

                r1 = s.request('get', url_file)
                r = s.get(r1.url, stream=True)
                if not r.ok:
                    raise IOError("Can't start download... [%s]" % url_file)

                file_size = int(r.headers['content-length'])

                LOG.info("Starting download on %s(%d bytes) ..." %
                         (file_name, file_size))

                with open(file_name, 'wb') as fp:
                    for chunk in r.iter_content(chunk_size=CHUNKS):
                        if chunk:
                            fp.write(chunk)
                    fp.flush()
                    os.fsync(fp)
                    if verbose:
                        LOG.info("\tDone!")

                for url_jpg in url_jpgs:
                    print(product_id, url_jpg)
                    wget.download(url_jpg, out= os.path.join(directory_path, product_name))


class download_SMAP(object):
    """
    The MODIS instrument is operating on both the Terra and Aqua spacecraft.
    It has a viewing swath width of 2,330 km and views the entire surface of the Earth every one to two days.
    Its detectors measure 36 spectral bands between 0.405 and 14.385 µm, and it acquires data at three spatial resolutions -- 250m, 500m, and 1,000m.

    :return:
    """

    def __init__(self, username, password):
        super(download_SMAP, self).__init__()

        self.username = username
        self.password = password
        self.product_shortname = None

    def fetch_datasets(self, download_type='ROI_polygon',
                       roi_polygon=None,
                       startdate=None,
                       enddate=None,
                       cloudcover_max=5,
                       product_shortname="SPL3SMP",
                       max_products=-1,
                       inverse_polygon_order=False):

        """Query NASA API for products

        See: https://modis.ornl.gov/data/modis_webservice.html

        """
        from cmr import GranuleQuery

        self.product_shortname = product_shortname
        self.roi_polygon = roi_polygon

        if download_type == 'ROI_polygon':
            if roi_polygon.split('.')[-1] == 'geojson':
                list_coords = from_geojson_to_list_coords(self.roi_polygon)
                #print(list_coords)

        else:
            raise RuntimeError("Unknown download type")

        if inverse_polygon_order:
            list_coords = list_coords[::-1]

        api = GranuleQuery().polygon(list_coords).short_name(self.product_shortname).temporal(startdate, enddate)

        n_produtes = api.hits()
        #print(f"{n_produtes} products found for these parameters")

        if max_products == -1:
            max_products = n_produtes

        self.list_products = api.get(limit=max_products)
        self.list_products_id = [ f["producer_granule_id"] for f in self.list_products]

    def download_files(self, list_product_ids,
                       directory_path='.',
                       verbose=True):
        """download the products"""

        import requests
        import sys
        import logging
        import wget

        LOG = logging.getLogger(__name__)
        OUT_HDLR = logging.StreamHandler(sys.stdout)
        OUT_HDLR.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        OUT_HDLR.setLevel(logging.INFO)
        LOG.addHandler(OUT_HDLR)
        LOG.setLevel(logging.INFO)


        CHUNKS = 65536

        with requests.Session() as s:
            s.auth = (self.username, self.password)

            for product_id in list_product_ids:

                if verbose:
                    print("Downloading prodcut:", product_id)
                index_product = self.list_products_id.index(product_id)

                product_name = product_id.split(".")[0]
                print(product_name)

                """Create the folder for the product"""
                try:
                    os.mkdir(os.path.join(directory_path, product_name), )
                except FileExistsError:
                    pass  # May need to check if the file is already downloaded

                product_links = self.list_products[index_product]["links"]

                url_jpgs = []
                url_file = None
                for p_link in product_links:
                    if "type" in p_link:
                        if p_link["type"] == 'application/x-hdfeos' and p_link["rel"] == 'http://esipfed.org/ns/fedsearch/1.1/data#' :

                            url_file = p_link["href"]
                        elif p_link["type"] == "image/jpeg":
                            if verbose:
                                print("The link object is", p_link)
                            url_jpgs.append(p_link["href"])

                assert url_file is not None, "URL file to download not found in Links attributes"

                file_name = os.path.join(directory_path, product_name, product_id, )
                print(url_file)
                r1 = s.request('get', url_file)
                r = s.get(r1.url, stream=True)
                if not r.ok:
                    raise IOError("Can't start download... [%s]" % url_file)

                try:
                    file_size = int(r.headers['content-length'])
                except KeyError:
                    file_size = 0

                LOG.info("Starting download on %s(%d bytes) ..." %
                         (file_name, file_size))

                with open(file_name, 'wb') as fp:
                    for chunk in r.iter_content(chunk_size=CHUNKS):
                        if chunk:
                            fp.write(chunk)
                    fp.flush()
                    os.fsync(fp)
                    if verbose:
                        LOG.info("\tDone!")

                for url_jpg in url_jpgs:
                    print(product_id, url_jpg)
                    wget.download(url_jpg, out= os.path.join(directory_path, product_name))


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

    def remove_nan(self):
        """
        TBD
        :return:
        """


class read_sentinel_2(object):

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.level = self.folder_path.split('/')[-1].split('.')[0].split('_')[1][-2:]
        if self.level == '2A':
            self.band_files = glob(self.folder_path + '/GRANULE' + '/*/IM*/*10*/*B*')
            for ext in ('*B05*', '*B06*', '*B07*', '*B8A*', '*B11*', '*B12*'):
                self.band_files.extend(glob(os.path.join(self.folder_path + '/GRANULE' + '/*/IM*/*20*/', ext)))
            for ext in ('*B01*', '*B09*', '*B10*'):
                self.band_files.extend(glob(os.path.join(self.folder_path + '/GRANULE' + '/*/IM*/*60*/', ext)))
            self.band_files = np.array(sorted(self.band_files))
            self.band_details = [file.split('_')[-2] for file in self.band_files]

        else:
            self.band_files = np.array(sorted(glob(self.folder_path + '/GRANULE' + '/*/IM*/*B*.jp2')))
            self.band_details = [file.split('_')[-1].split('.')[0] for file in self.band_files]
        im = gdal.Open(self.band_files[1])
        # arf_base = im.ReadAsArray()
        # self.img_shp = arf_base.shape
        self.meta_data = im.GetMetadata()
        self.num_bands = len(self.band_files)
        self.save_folder = self.folder_path
        self.AOI = None
        self.data = None
        self.data_dictionary = None

    def get_data(self, AOI='all', resample=False, resize_raster_source='B02',
                 interpolation=Resampling.cubic_spline, save_as_npy=False):
        """
        :return: (a dictionary of 13 bands),(single array of dimension (13,x_img,y_img) in the 'order of bands')
        """

        if AOI == 'all':
            self.band_files_new = self.band_files
            print('no AOI selected')
        else:
            # check for existing clipped raster files in tiff for given aoi
            aoi_name = AOI.split('/')[-1].split('.')[0]
            self.band_files_new = np.array(sorted(glob(self.save_folder + '/' + aoi_name + '*.tiff')))

        if len(self.band_files_new) == 0:
            if AOI != 'all':
                start = time.time()
                for source_raster in self.band_files:
                    clip_to_aoi(source_raster, AOI, self.save_folder)

                print('clipped to AOI in: ', time.time() - start, ' seconds')
                self.band_files_new = np.array(sorted(glob(self.save_folder + '/' + aoi_name + '*.tiff')))

        data_dictionary = {}
        print("loading data...")
        for band in self.band_files_new:
            if self.level =='2A':
                name = '_'.join(band.split('_')[-3:-1])
                key = band.split('_')[-2]
            else:
                name = band.split('/')[-1].split('.')[0]
                key = band.split('_')[-1].split('.')[0]
            print('loading ',name)
            arf = gdal.Open(band)
            data = arf.ReadAsArray()
            if len(data.shape)==3:
                data = data[0]

            if save_as_npy:
                self.save_as_npy(data, save_path=self.save_folder + '/' + name)

            data_dictionary[key] = data
        self.data_dictionary = data_dictionary
        print("all 13 bands loaded")
        if resample:
            print("resampling data started...")
            start = time.time()
            self.data_resampled = self.sentinel_2_remap(ref_raster=resize_raster_source, interpolation=interpolation)
            print('resampled in: ', time.time() - start, ' seconds')
        else:
            print('resampling not selected ')
            self.data_resampled =[]
        return self.data_dictionary, self.data_resampled

    def sentinel_2_remap(self, ref_raster='B02', interpolation=Resampling.cubic_spline):
        """
        Upscaling only
        :param interpolation:Resampling.cubic_spline,Resampling.cubic,Resampling.average,
                             Resampling.bilinear, Resampling.gauss, Resampling.lanczos
        :return: array(13,x_img,y_img)
        """
        band_names = sorted(self.data_dictionary.keys())
        num_bands = len(band_names)
        rows, cols = self.data_dictionary[ref_raster].shape
        data_resampled = np.zeros((num_bands, rows, cols))

        for i in range(len(band_names)):
            if (rows, cols) == self.data_dictionary[band_names[i]].shape:
                data_resampled[i, :, :] = self.data_dictionary[band_names[i]]
            else:
                with rasterio.open(self.band_files_new[i]) as arf:
                    ar = arf.read(out_shape=(arf.count, rows, cols), resampling=interpolation)
                    data_resampled[i, :, :] = ar[0]
            print(band_names[i],' done')

        return data_resampled

    def remove_nan(self):
        """
        TBD
        :return:
        """

    def save_as_npy(self, data, save_path=os.getcwd() + '/data'):
        np.save(save_path, self.data)
        print('dataset saved in .npy format at this location:', save_path)

    @staticmethod
    def get_ndvi(data, nir_index=7, red_index=3):
        """
        Normalized difference vegetation index
        NDVI = (b_nir - b_red)/(b_nir + b_red)
        :return: NDVI values for each pixel
        """
        ndvi = (data[nir_index,:, :] - data[red_index,:, :]) / (data[nir_index,:, :] + data[red_index,:, :])

        return ndvi

    @staticmethod
    def get_ndwi(data, nir_index=7, green_index=2, swir_index=10, option=1):
        """
        Normalized difference water index

        option 1:
        NDWI = (b_green - b_nir)/(b_nir + b_green)

        option 2:
        NDWI = (b_nir - b11_swir)/(b_nir + b11_swir)
        :return: NDWI values for each pixel
        """
        if option == 1:
            ndwi = (data[green_index,:, :] - data[nir_index,:, : ]) / (data[nir_index,:, :] + data[green_index,:, :])
        elif option == 2:
            """
            needs resampling band 11 SWIR to 10x10 or band 8 NIR to 20x20
            """
            ndwi = (data[nir_index,:, :] - data[swir_index,:, :]) / (data[nir_index,:, :] + data[swir_index,:, :])

        return ndwi

    @staticmethod
    def get_lai(data):
        """
        Leaf Area Index
        source  : Clevers, J.G.P.W.; Kooistra, L.; van den Brande, M.M.M. ,
        Using Sentinel-2 Data for Retrieving LAI and Leaf and Canopy Chlorophyll Content of a Potato Crop. ,
        Remote Sens. 2017, 9, 405.

        :param data:
        :return:
        """
        return NotImplementedError


def from_geojson_to_list_coords(filename):
    """Part the geojson file and return the list of points"""
    geo_json_roi = read_geojson(filename)


    if geo_json_roi["type"] == 'FeatureCollection':
        geo_json_features = geo_json_roi["features"]
        n_features = len(geo_json_features)

        assert n_features == 1, "The number of features must be 1"

        geo_json_feature = geo_json_features[0]["geometry"]

        assert geo_json_feature["type"] == "Polygon" , "Feature types other than polygons is not yet possible"

        list_coordinates = geo_json_feature["coordinates"][0]

    else:
        raise RuntimeError("Unknown Geojson type")


    return list_coordinates

