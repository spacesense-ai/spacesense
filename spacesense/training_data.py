"""
To assist in building training dataset

"""
import os
import urllib
import zipfile
import glob
import shapefile
from shapely.geometry import Polygon
from os.path import exists
import numpy as np
from glob import glob
import time
import gdal
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class manage():
    def __init__(self,folder='training_files/'):
        self.filetype = 'shp'
        self.folder = os.path.abspath(folder)

    def display_shp_polygon(self,file):
        if file.split('.')[-1] == 'shp':
            sf = shapefile.Reader(folder+'/'+file)
            polygon = Polygon(sf.shape().points)
            x,y = polygon.exterior.xy
            plt.plot(x,y)
            plt.xlim(sf.shape().bbox[0], sf.shape().bbox[2])
            plt.show()


class EuroSAT(object):
    def __init__(self):
        """
        Dataset source and reference: https://github.com/phelber/eurosat
        Same copyright and license applies as described at the dataset source.
        Band List in order:
        B01 - Aerosols B02 - Blue B03 - Green B04 - Red
        B05 - Red edge 1 B06 - Red edge 2 B07 - Red edge 3
        B08 - NIR B08A - Red edge 4 B09 - Water vapor
        B10 - Cirrus B11 - SWIR 1 B12 - SWIR 2
        """
        self.data_path_all_bands = 'data/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        self.data_path_rgb = 'data/2750'
        self.n_labels = 10
        self.label_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                            'PermanentCrop', 'Residential', 'River', 'SeaLake']

        self.n_samples = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
        self.info = {'total sample size': 27000,
                     'labels': {}}
        for i in range(len(self.label_names)):
            self.info['labels'][self.label_names[i]] = self.n_samples[i]

    def download_all_bands(self, save_as_npy=False):
        """
        :return: folder with training data (2.9GB)
        """
        if exists(self.data_path_all_bands):
            print('dataset is already available at: ', os.path.abspath(self.data_path_all_bands))

        else:
            os.system("mkdir data")
            os.system("wget http://madm.dfki.de/files/sentinel/EuroSATallBands.zip -P data/")
            file_name = 'data/EuroSATallBands.zip'

            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall('data/')

            self.data_path_all_bands = 'data/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
            print('EuroSAT all bands data downloaded !')

    def download_rgb(self, save_as_npy=False):
        """
        :return: folder with RGB training data (93MB)
        """
        if exists(self.data_path_rgb):
            print('dataset is already available at: ', os.path.abspath(self.data_path_rgb))
        else:
            os.system("mkdir data")
            os.system("wget madm.dfki.de/files/sentinel/EuroSAT.zip -P data/")
            file_name = 'data/EuroSAT.zip'

            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall('data/')

            self.data_path_rgb = 'data/2750'
            print('EuroSAT RGB data downloaded !')

    def fetch_data(self, datatype='all_bands', labels='all', row_type='2D'):
        """

        :param type:
        :param labels:
        :return: numpy array
        """
        if datatype == 'all_bands':
            paths = sorted(glob(self.data_path_all_bands + '/*'))
            if exists(self.data_path_all_bands):
                X, y = self.__load_dataset__(datatype='all_bands',labels=labels,row_type=row_type)

            else:
                print('dataset is not available')
                print('to download the dataset, choose one of the two options:')
                print('EuroSAT.download_all_bands()')
                print('or')
                print('EuroSAT.download_rgb()')

        elif datatype == 'rgb':
            paths = sorted(glob(self.data_path_rgb + '/*'))
            if exists(self.data_path_rgb):
                X, y = self.__load_dataset__(datatype='rgb',labels=labels,row_type=row_type)

            else:
                print('dataset is not available')
                print('to download the dataset, choose one of the two options:')
                print('EuroSAT.download_all_bands()')
                print('or')
                print('EuroSAT.download_rgb()')
        return X, y

    def __load_dataset__(self, datatype,labels, row_type):

        if datatype == 'all_bands':
            data_path = self.data_path_all_bands
            n_bands = 13
        elif datatype == 'rgb':
            data_path = self.data_path_rgb
            n_bands = 3

        start = time.time()
        if labels == 'all':
            if row_type == '1D':
                x = np.zeros((self.info['total sample size'] * 64 * 64, n_bands))
                y = np.zeros(self.info['total sample size'] * 64 * 64)
            elif row_type == '2D':
                x = np.zeros((self.info['total sample size'], n_bands, 64, 64))
                y = np.zeros(self.info['total sample size'])

            labels = self.label_names

        else:
            sample_size = np.sum([self.info['labels'][name] for name in labels])
            if row_type == '1D':
                x = np.zeros((sample_size * 64 * 64, n_bands))
                y = np.zeros(sample_size * 64 * 64)
            elif row_type == '2D':
                x = np.zeros((sample_size, n_bands, 64, 64))
                y = np.zeros(sample_size)

        img_paths = {}
        i = 0
        j = 0
        for label in labels:
            folder_path = os.path.join(data_path, label)
            img_paths[label] = glob(folder_path + '/*')
            folder_path = os.path.join(data_path, label)
            img_paths[label] = glob(folder_path + '/*')
            for img in img_paths[label]:
                data = gdal.Open(img).ReadAsArray()
                if row_type == '1D':
                    x_img = self.get_data(data)
                    n = x_img.shape[0]
                    x[i:i + n] = x_img
                    y[i:i + n] = self.label_names.index(label)
                    i += n
                elif row_type=='2D':
                    x[j] = data
                    y[j] = self.label_names.index(label)
                    j += 1

            print(label + ' data loaded')
        print('time taken: %d seconds' % (time.time() - start))
        print('x.shape: ', x.shape, 'y.shape: ', y.shape)
        return x, y

    def display_sample(self, labels='all'):
        if labels == 'all':
            labels = self.label_names
        for label in labels:
            if label not in self.label_names:
                print('unavailable label')
                print('choose only available label names from "self.label_names" ')
                break
            else:
                i = np.random.choice(range(self.info['labels'][label]))
                folder_path = os.path.join(self.data_path_rgb, label)
                images = glob(folder_path + '/*')
                fig = plt.figure(i)
                fig.suptitle(label)
                plt.imshow(mpimg.imread(images[i]))
                plt.show()

    @staticmethod
    def get_data(x):
        if len(x.shape) == 3:
            if (x.shape[2] < x.shape[0]) and (x.shape[2] < x.shape[1]):
                X = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            elif (x.shape[0] < x.shape[1]) and (x.shape[0] < x.shape[2]):
                X = x.reshape(x.shape[1] * x.shape[2], x.shape[0])
                X = X.T
        else:
            X = x
        return X

    @staticmethod
    def save_as_npy(dataset, save_folder=os.getcwd(), file_name='data'):
        path = os.path.join(save_folder, file_name)
        np.save(path, dataset)
        print('dataset saved in .npy format at this location:', save_folder)


