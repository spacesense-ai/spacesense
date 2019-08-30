
import datetime
from datetime import date

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spacesense.datasets import download_SMAP


def test_download_smap():
    """Test the sentinel API by looking for the observation over Paris in 2016.
    We espect one observation

    """
    import os
    username = os.environ["MODIS_LOGIN"]
    psswrd = os.environ["MODIS_PWD"]


    d = download_SMAP(username, psswrd)

    geojson_file = os.path.abspath("./examples/Paris-15.geojson")

    d.fetch_datasets(roi_polygon=geojson_file, startdate=date(2019, 1, 1), enddate=date(2019,6,30), inverse_polygon_order=True)

    # print(d.list_products)

    assert len(d.list_products) == 170


    id =  d.list_products_id[1]
    assert id == 'SMAP_L3_SM_P_20190102_R16022_001.h5'

    data = d.list_products[0]

    dict_refecrance = {'producer_granule_id': 'SMAP_L3_SM_P_20190101_R16022_001.h5',
 'boxes': ['-85.0445023 -180 85.0445023 180'],
 'time_start': '2019-01-01T00:00:00.000Z',
 'updated': '2019-07-12T16:31:30.636Z',
 'dataset_id': 'SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture V005',
 'data_center': 'NSIDC_ECS',
 'title': 'SC:SPL3SMP.005:145049886',
 'coordinate_system': 'CARTESIAN',
 'time_end': '2019-01-01T23:59:59.000Z',
 'id': 'G1581610786-NSIDC_ECS',
 'original_format': 'ISO-SMAP',
 'granule_size': '27.0171604156',
 'browse_flag': False,
 'collection_concept_id': 'C1522347056-NSIDC_ECS',
 'online_access_flag': True,
 'links': [{'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'type': 'application/x-hdfeos',
   'hreflang': 'en-US',
   'href': 'https://n5eil01u.ecs.nsidc.org/DP4/SMAP/SPL3SMP.005/2019.01.01/SMAP_L3_SM_P_20190101_R16022_001.h5'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'type': 'application/x-hdfeos',
   'hreflang': 'en-US',
   'href': 'https://n5eil01u.ecs.nsidc.org/opendap/DP4/SMAP/SPL3SMP.005/2019.01.01/SMAP_L3_SM_P_20190101_R16022_001.h5'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/documentation#',
   'type': 'text/plain',
   'hreflang': 'en-US',
   'href': 'https://n5eil01u.ecs.nsidc.org/DP1/AMSA/QA.001/2019.01.02/SMAP_L3_SM_P_20190101_R16022_001.qa'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'type': 'text/xml',
   'hreflang': 'en-US',
   'href': 'https://n5eil01u.ecs.nsidc.org/DP4/SMAP/SPL3SMP.005/2019.01.01/SMAP_L3_SM_P_20190101_R16022_001.h5.iso.xml'},
  {'inherited': True,
   'length': '0.0KB',
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'https://nsidc.org/daac/subscriptions.html'},
  {'inherited': True,
   'length': '0.0KB',
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL3SMP.005/'},
  {'inherited': True,
   'length': '0.0KB',
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'https://search.earthdata.nasa.gov/search/granules?p=C1522347056-NSIDC_ECS&m=-33.046875!19.96875!1!1!0!0%2C2&tl=1518544294!4!!&q=SPL3SMP'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'hreflang': 'en-US',
   'href': 'https://doi.org/10.5067/ZX7YX2Y2LHEB'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/documentation#',
   'hreflang': 'en-US',
   'href': 'https://doi.org/10.5067/ZX7YX2Y2LHEB'}]}

    assert dict_refecrance == data

    # for k, v in dict_refecrance.items():
    #     #print(k)
    #
    #     if not k in data:
    #         raise RuntimeError("Error this", k)
    #
    #     if data[k] != v:
    #         raise RuntimeError("Error that", data[k])




if __name__=="__main__":
    test_download_modis()