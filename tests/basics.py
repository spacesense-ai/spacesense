
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date

from spacesense.datasets import download_sentinel

def test_download_sentinel():

    username = "antoinetavant"
    psswrd = "sdr2sebzaf845dg"

    d = download_sentinel(username, psswrd)

    geojson_file = "./examples/Paris-15.geojson"

    d.sentinel(roi_polygon=geojson_file, startdate=date(2016, 6, 15), enddate=date(2016, 6, 26), )
    print(d.list_products)


