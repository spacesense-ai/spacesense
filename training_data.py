"""


"""
import shapefile
import matplotlib
import os
from shapely.geometry import Point,Polygon,LineString,MultiPolygon

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









