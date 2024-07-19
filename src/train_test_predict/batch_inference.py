import ee
import sys
import geemap
import torch
import rasterio
import os
import gc
import json

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from data.fusion_dataset import FusionDataset
from models.lit_model_lf import LitModelLateFusion
from utils.generate_grid import generate_grid
from utils.export_tile import export_tile
from utils.is_export_done import is_export_done
from utils.download_gee_asset import download_gee_asset
from utils.delete_local_files import delete_local_files
from utils.batch_delete_assets import batch_delete_assets

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
    print("s1_preproc function imported successfully", flush=True)
except ImportError as e:
    print("Failed to import the module:", e, flush=True)
    sys.exit(1)

# set seed for reproducibility
seed_everything(42, workers=True)

# load trained model
model = LitModelLateFusion
# model.load_state_dict(torch.load('models/fusion_full.pth'))
model = model.load_from_checkpoint('models/checkpoints/fusion_pretrained_trial11-epoch=43-val_f1score=0.84.ckpt')
model.eval()
model.freeze()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# define AOI and convert to EE object
aoi_path = "/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/study_area/mwenga_wgs84.geojson"
with open(aoi_path, 'r') as f:
    geojson_dict = json.load(f)
ee_object = geemap.geojson_to_ee(geojson_dict)

# create grid over AOI
grid_tiles = generate_grid(ee_object, 4.77, 1000, 1000)

output_dir = 'models/predictions/inference'
os.makedirs(output_dir, exist_ok=True)

# re-initialize EE
try:
    ee.Initialize(project='asm-drc')
except Exception as e:
    print(f"Failed to reinitialize Earth Engine: {e}", flush=True)
    sys.exit(1)

BATCH_SIZE = 10  # set tiles batch size
tile_info_list = [(i, ee.Feature(tile)) for i, tile in enumerate(grid_tiles.toList(grid_tiles.size()).getInfo())]

# processing loop
for batch_start in range(0, len(tile_info_list), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(tile_info_list))
    batch = tile_info_list[batch_start:batch_end]
    
    for tile_info in batch:
        i, tile = tile_info
        ee.Initialize(project='asm-drc')
        print(f"Processing tile {i}", flush=True)

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
            print(f"Skipping tile {i} due to failed or incomplete export", flush=True)
            continue

        # downlaod GEE assets as GeoTIFF
        nicfi_local_path = f"/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/inference_dataset/inference_gee/images/planet/nicfi_tile_{i}.tif"
        s1_local_path = f"/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/inference_dataset/inference_gee/images/s1/s1_tile_{i}.tif"
        download_gee_asset(f'projects/asm-drc/assets/{nicfi_asset_id}', tile, nicfi_local_path)
        download_gee_asset(f'projects/asm-drc/assets/{s1_asset_id}', tile, s1_local_path)

        # read data with dataset class
        data_dir = '/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/inference_dataset/inference_gee'
        dataset = FusionDataset(
            root_dir=data_dir,
            planet_normalization=True,
            s1_normalization=True,
            is_inference=True,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for _, batch in enumerate(dataloader):
            # clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

            # get data from batch
            planet_input, s1_input = batch[:2]
            file_path = dataset.planet_dataset.dataset[0]

            # make prediction
            with torch.no_grad():
                logits = model(planet_input.to(model.device), s1_input.to(model.device))
            probs = torch.sigmoid(logits)
            prediction_binary = (probs > 0.3).float()
            prediction_np = prediction_binary.cpu().numpy().squeeze()
            probs_np = probs.cpu().numpy().squeeze()

            with rasterio.open(file_path) as src:
                crs = src.crs
                transform = src.transform

            output_path = os.path.join(output_dir, f"prediction_{i}.tif")
            
            # save prediction
            with rasterio.open(output_path, 'w', driver='GTiff',
                            height=prediction_np.shape[0],
                            width=prediction_np.shape[1],
                            count=1, dtype='float32',
                            crs=crs,
                            transform=transform) as dst:
                dst.write(prediction_np, 1)
                dst.write(probs_np, 2)
            print(f'Saved prediction to {output_path}', flush=True)

            # delete local files and GEE assets
            asset_list = []
            asset_list.append(nicfi_asset_id)
            asset_list.append(s1_asset_id)
            batch_delete_assets(asset_list)
            delete_local_files([nicfi_local_path, s1_local_path])

            print(f"Tile {i} processed successfully", flush=True)

    print(f"Processed batch {batch_start // BATCH_SIZE + 1} of {len(tile_info_list) // BATCH_SIZE + 1}", flush=True)
