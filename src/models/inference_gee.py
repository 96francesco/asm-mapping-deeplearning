import ee
import sys
import geemap
import torch
import rasterio
import os
import gc
import requests
import json
import time

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from data.fusion_dataset import FusionDataset
from data.planet_dataset_normalization import linear_norm_global_percentile as planet_norm
from data.s1_dataset_normalization import linear_norm_global_percentile as s1_norm
from models.lit_model_lf import LitModelLateFusion

from utils.generate_grid import generate_grid
from utils.export_tile import export_tile
from utils.is_export_done import is_export_done
from utils.download_gee_asset import download_gee_asset
from utils.delete_gee_asset import delete_gee_asset
from utils.delete_local_files import delete_local_files


# authenticate and initialize GEE
ee.Authenticate()
ee.Initialize(project='asm-drc')

# path to script
script_dir = os.path.dirname(os.path.abspath(__file__))

# path to the utils directory
utils_dir = os.path.join(script_dir, '..', 'utils')

# path to the gee_s1_ard/python-api directory
repo_path = os.path.join(utils_dir, 'gee_s1_ard', 'python-api')

# add gee_s1_ard/python-api to the Python path
sys.path.append(repo_path)

# import the Sentinel-1 preprocessing function
try:
    from wrapper import s1_preproc
    print("s1_preproc function imported successfully")
except ImportError as e:
    print("Failed to import the module:", e)
    sys.exit(1)


# set seed for reproducibility
seed_everything(42, workers=True)

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# load trained model
model = LitModelLateFusion()
model.load_state_dict(torch.load('models/fusion_full.pth'))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# define AOI and convert to EE object
aoi_path = "/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/study_area/north_kivu_wgs84.geojson"
with open(aoi_path, 'r') as f:
    geojson_dict = json.load(f)
ee_object = geemap.geojson_to_ee(geojson_dict)

# create grid over AOI
print(type(ee_object))
grid_tiles = generate_grid(ee_object, 4.77, 375, 375)

# processing loop
output_dir = 'models/predictions/inference'
os.makedirs(output_dir, exist_ok=True)

# re-initialize EE
# if this is not done, an error raises
try:
        ee.Initialize(project='asm-drc')
except Exception as e:
    print(f"Failed to reinitialize Earth Engine: {e}")
    sys.exit(1)
for i, tile in enumerate(grid_tiles.toList(grid_tiles.size()).getInfo()):
    tile = ee.Feature(tile)
    print(f"Processing tile {i + 1}/{grid_tiles.size().getInfo()}")

    # fetch and export imagery
    nicfi_asset_id = f"nicfi_tile_{i}"
    s1_asset_id = f"s1_tile_{i}"
    nicfi_task = export_tile(
        ee.ImageCollection('projects/planet-nicfi/assets/basemaps/africa')\
            .filterDate('2023-01-01', '2023-06-30')\
            .median(),
        tile,
        nicfi_asset_id
    )
    params = {
        'APPLY_BORDER_NOISE_CORRECTION': True,
        'APPLY_TERRAIN_FLATTENING': True,
        'APPLY_SPECKLE_FILTERING': False,
        'POLARIZATION': 'VVVH',
        'ORBIT': 'DESCENDING',
        'START_DATE': '2023-01-01',
        'STOP_DATE': '2023-06-30',
        'ROI': ee_object.geometry(),
        'PLATFORM_NUMBER': 'A',
        'ORBIT_NUM': None,
        'SPECKLE_FILTER_FRAMEWORK': 'MULTI',
        'SPECKLE_FILTER': 'GAMMA MAP',
        'SPECKLE_FILTER_KERNEL_SIZE': 3,
        'SPECKLE_FILTER_NR_OF_IMAGES': 10,
        'DEM': ee.Image('USGS/SRTMGL1_003'),
        'TERRAIN_FLATTENING_MODEL': 'VOLUME',
        'TERRAIN_FLATTENING_ADDITIONAL_LAYOVER_SHADOW_BUFFER': 0,
        'FORMAT': 'DB',
        'CLIP_TO_ROI': False,
        'SAVE_ASSET': False,
        'ASSET_ID': None
    }
    processed_s1_collection = s1_preproc(params)
    s1_images_preprocessed = processed_s1_collection\
        .select(['VV', 'VH'])\
        .mean()
    s1_task = export_tile(
        s1_images_preprocessed,
        tile,
        s1_asset_id
    )

    # wait for tasks to complete
    if not is_export_done(nicfi_task) or not is_export_done(s1_task):
        print(f"Skipping tile {i + 1} due to failed or incomplete export")
        continue

    # downlaod GEE assets as GeoTIFF
    nicfi_local_path = f"/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/inference_dataset/inference_gee/planet/nicfi_tile_{i}.tif"
    s1_local_path = f"/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/inference_dataset/inference_gee/s1/s1_tile_{i}.tif"
    download_gee_asset(f'projects/asm-drc/assets/{nicfi_asset_id}', tile, nicfi_local_path)
    download_gee_asset(f'projects/asm-drc/assets/{s1_asset_id}', tile, s1_local_path)

    # TODO:
    # 1. Read downloaded images with FusionDataset
    # 2. Implement model inference
    # 3. Save preidctions as GeoTIFF



