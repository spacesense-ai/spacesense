
import datetime
from datetime import date

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spacesense.datasets import download_modis


def test_download_modis():
    """Test the sentinel API by looking for the observation over Paris in 2016.
    We espect one observation

    """
    import os
    username = os.environ["MODIS_LOGIN"]
    psswrd = os.environ["MODIS_PWD"]


    d = download_modis(username, psswrd)

    geojson_file = os.path.abspath("./examples/Paris-15.geojson")

    d.fetch_datasets(roi_polygon=geojson_file, startdate=date(2019, 1, 1), enddate=date(2019,6,30), inverse_polygon_order=True)

    # print(d.list_products)

    assert len(d.list_products) == 24


    id =  d.list_products_id[0]
    assert id == 'MOD15A2H.A2018361.h18v04.006.2019009093326.hdf'

    data  = d.list_products[0]

    dict_refecrance = {'producer_granule_id': 'MOD15A2H.A2018361.h18v04.006.2019009093326.hdf',
 'time_start': '2018-12-27T00:00:00.000Z',
 'cloud_cover': '17.0',
 'updated': '2019-01-09T03:45:32.705Z',
 'dataset_id': 'MODIS/Terra Leaf Area Index/FPAR 8-Day L4 Global 500m SIN Grid V006',
 'data_center': 'LPDAAC_ECS',
 'title': 'SC:MOD15A2H.006:2326164933',
 'coordinate_system': 'GEODETIC',
 'day_night_flag': 'DAY',
 'time_end': '2019-01-03T23:59:59.000Z',
 'id': 'G1582757469-LPDAAC_ECS',
 'original_format': 'ECHO10',
 'granule_size': '8.37003',
 'browse_flag': True,
 'polygons': [['39.8144151 13.0379033 49.9989722 15.5723927 50.006969 -0.0086748 39.8197706 0.0001314 39.8144151 13.0379033']],
 'collection_concept_id': 'C203669720-LPDAAC_ECS',
 'online_access_flag': True,
 'links': [{'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'type': 'application/x-hdfeos',
   'title': 'This file may be downloaded directly from this link',
   'hreflang': 'en-US',
   'href': 'https://e4ftl01.cr.usgs.gov//MODV6_Cmp_A/MOLT/MOD15A2H.006/2018.12.27/MOD15A2H.A2018361.h18v04.006.2019009093326.hdf'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/documentation#',
   'type': 'text/html',
   'title': 'This file may be accessed using OPeNDAP directly from this link (OPENDAP DATA)',
   'hreflang': 'en-US',
   'href': 'https://opendap.cr.usgs.gov/opendap/hyrax//MODV6_Cmp_A/MOLT/MOD15A2H.006/2018.12.27/MOD15A2H.A2018361.h18v04.006.2019009093326.hdf'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/browse#',
   'type': 'image/jpeg',
   'title': 'This Browse file may be downloaded directly from this link (BROWSE)',
   'hreflang': 'en-US',
   'href': 'https://e4ftl01.cr.usgs.gov//WORKING/BRWS/Browse.001/2019.01.09/BROWSE.MOD15A2H.A2018361.h18v04.006.2019009093326.1.jpg'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/browse#',
   'type': 'image/jpeg',
   'title': 'This Browse file may be downloaded directly from this link (BROWSE)',
   'hreflang': 'en-US',
   'href': 'https://e4ftl01.cr.usgs.gov//WORKING/BRWS/Browse.001/2019.01.09/BROWSE.MOD15A2H.A2018361.h18v04.006.2019009093326.2.jpg'},
  {'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'type': 'text/xml',
   'title': 'This Metadata file may be downloaded directly from this link (EXTENDED METADATA)',
   'hreflang': 'en-US',
   'href': 'https://e4ftl01.cr.usgs.gov//MODV6_Cmp_A/MOLT/MOD15A2H.006/2018.12.27/MOD15A2H.A2018361.h18v04.006.2019009093326.hdf.xml'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'hreflang': 'en-US',
   'href': 'https://doi.org/10.5067/MODIS/MOD15A2H.006'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'https://e4ftl01.cr.usgs.gov/MOLT/MOD15A2H.006/'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'http://earthexplorer.usgs.gov/'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'https://search.earthdata.nasa.gov/search?q=MOD15A2H+V006'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'hreflang': 'en-US',
   'href': 'https://opendap.cr.usgs.gov/opendap/hyrax/MOD15A2H.006/contents.html'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/metadata#',
   'hreflang': 'en-US',
   'href': 'https://lpdaac.usgs.gov/'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/documentation#',
   'hreflang': 'en-US',
   'href': 'https://lpdaac.usgs.gov/documents/2/mod15_user_guide.pdf'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/documentation#',
   'hreflang': 'en-US',
   'href': 'https://lpdaac.usgs.gov/documents/90/MOD15_ATBD.pdf'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/documentation#',
   'hreflang': 'en-US',
   'href': 'https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MOD15A2H'},
  {'inherited': True,
   'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
   'hreflang': 'en-US',
   'href': 'https://lpdaac.usgs.gov/tools/appeears/'}]}

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