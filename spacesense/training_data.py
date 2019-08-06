"""
To assist in building training dataset


"""
import os
import zipfile
import glob
import shapefile
from shapely.geometry import Polygon




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
        """
        self.data_path_all_bands = 'data/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        self.data_path_rgb = 'data/2750'
        self.n_labels = 10
        self.label_names = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial', 'Pasture',
                            'PermanentCrop','Residential', 'River','SeaLake']

        self.n_samples = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
        self.info = {'total sample size': 27000,
                     'labels':{}}
        for i in range(len(names)):
            self.info['labels'][self.label_names[i]]= self.n_samples[i]

    def download_all_bands(self):
        """
        :return: folder with training data (2.9GB)
        """
        os.system("mkdir data")
        os.system("wget http://madm.dfki.de/files/sentinel/EuroSATallBands.zip -P data/")
        file_name = 'data/EuroSATallBands.zip'

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall('data/')

        self.data_path_all_bands = 'data/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        print('EuroSAT all bands data downloaded !')

    def download_rgb(self):
        """
        :return: folder with RGB training data (93MB)
        """
        os.system("mkdir data")
        os.system("wget madm.dfki.de/files/sentinel/EuroSAT.zip -P data/")
        file_name = 'data/EuroSAT.zip'

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall('data/')

        self.data_path_rgb = 'data/2750'
        print('EuroSAT RGB data downloaded !')

    def get_all_bands(self):


    def get_rgb(self):







