import ee
import requests

def download_gee_asset(asset_id, ee_object, local_path):
      url = ee.Image(asset_id).getDownloadURL({'scale': 4.77, 'region': ee_object.geometry()})
      response = requests.get(url)
      with open(local_path, 'wb') as f:
            f.write(response.content)