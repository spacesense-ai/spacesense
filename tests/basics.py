
import datetime
from datetime import date

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spacesense.datasets import download_sentinel


def test_download_sentinel():
    """Test the sentinel API by looking for the observation over Paris in 2016.
    We espect one observation

    """

    username = "antoinetavant"
    psswrd = "sdr2sebzaf845dg"

    d = download_sentinel(username, psswrd)

    geojson_file = os.path.abspath("./examples/Paris-15.geojson")

    d.sentinel(roi_polygon=geojson_file, startdate=date(2016, 6, 15), enddate=date(2016, 6, 26), )

    # print(d.list_products)

    assert len(d.list_products) == 1

    assert d.list_products[0] == ('be8aafbe-1a0b-414d-8bcc-ae6c1f616ce9',
  {'title': 'S2A_MSIL1C_20160623T105652_N0204_R094_T31UDQ_20160623T105858',
   'link': "https://scihub.copernicus.eu/dhus/odata/v1/Products('be8aafbe-1a0b-414d-8bcc-ae6c1f616ce9')/$value",
   'link_alternative': "https://scihub.copernicus.eu/dhus/odata/v1/Products('be8aafbe-1a0b-414d-8bcc-ae6c1f616ce9')/",
   'link_icon': "https://scihub.copernicus.eu/dhus/odata/v1/Products('be8aafbe-1a0b-414d-8bcc-ae6c1f616ce9')/Products('Quicklook')/$value",
   'summary': 'Date: 2016-06-23T10:56:52.027Z, Instrument: MSI, Mode: , Satellite: Sentinel-2, Size: 636.33 MB',
   'datatakesensingstart': datetime.datetime(2016, 6, 23, 10, 56, 52, 27000),
   'beginposition': datetime.datetime(2016, 6, 23, 10, 56, 52, 27000),
   'endposition': datetime.datetime(2016, 6, 23, 10, 56, 52, 27000),
   'ingestiondate': datetime.datetime(2018, 10, 24, 9, 52, 59, 957000),
   'orbitnumber': 5239,
   'relativeorbitnumber': 94,
   'cloudcoverpercentage': 4.5617,
   'gmlfootprint': '<gml:Polygon srsName="http://www.opengis.net/gml/srs/epsg.xml#4326" xmlns:gml="http://www.opengis.net/gml">\n   <gml:outerBoundaryIs>\n      <gml:LinearRing>\n         <gml:coordinates>49.64443007346099,1.614272368185704 48.6570207750582,1.641546225080212 48.66495515183609,3.132551327123652 49.652643868751746,3.135213595780942 49.64443007346099,1.614272368185704</gml:coordinates>\n      </gml:LinearRing>\n   </gml:outerBoundaryIs>\n</gml:Polygon>',
   'format': 'SAFE',
   'instrumentshortname': 'MSI',
   'sensoroperationalmode': 'INS-NOBS',
   'instrumentname': 'Multi-Spectral Instrument',
   'footprint': 'MULTIPOLYGON (((1.614272368185704 49.64443007346099, 1.641546225080212 48.6570207750582, 3.132551327123652 48.66495515183609, 3.135213595780942 49.652643868751746, 1.614272368185704 49.64443007346099)))',
   's2datatakeid': 'GS2A_20160623T105652_005239_N02.04',
   'platformidentifier': '2015-028A',
   'orbitdirection': 'DESCENDING',
   'platformserialidentifier': 'Sentinel-2A',
   'processingbaseline': '02.04',
   'processinglevel': 'Level-1C',
   'producttype': 'S2MSI1C',
   'platformname': 'Sentinel-2',
   'size': '636.33 MB',
   'tileid': '31UDQ',
   'hv_order_tileid': 'UQ31D',
   'filename': 'S2A_MSIL1C_20160623T105652_N0204_R094_T31UDQ_20160623T105858.SAFE',
   'identifier': 'S2A_MSIL1C_20160623T105652_N0204_R094_T31UDQ_20160623T105858',
   'uuid': 'be8aafbe-1a0b-414d-8bcc-ae6c1f616ce9'})

