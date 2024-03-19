import torch
import rasterio
import os

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from data.fusion_dataset import FusionDataset
from data.planet_dataset_normalization import linear_norm_global_minmax as planet_norm
from data.s1_dataset_normalization import global_standardization as s1_norm
from models.lit_model_fusion import LitModelBinaryLateFusion

# set seed for reproducibility
seed_everything(42, workers=True)

# load trained model
model_checkpoint_path = 'models/checkpoints/fusion-split-0.ckpt'
model = LitModelBinaryLateFusion.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# define dataset
dataset_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/full_dataset'
dataset = FusionDataset(root_dir=dataset_dir,
                        train=False,
                        is_inference=True,
                        planet_normalization=planet_norm,
                        s1_normalization=s1_norm)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# load CRS
reference_image_path = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/full_dataset/images/planet/nicfi_0.tif'
with rasterio.open(reference_image_path) as ref:
    crs = ref.crs
    transform = ref.transform

output_folder = 'models/predictions/output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i, batch in enumerate(dataloader):
      planet_input, s1_input = batch[:2] 
      file_path = dataset.planet_dataset.dataset[i][0]
      
      with torch.no_grad():
            output = model(planet_input.to(model.device), s1_input.to(model.device))
      prediction = torch.sigmoid(output)
      
      # convert probabilities to classes
      prediction_binary = (prediction > 0.6).float()

      prediction_np = prediction_binary.cpu().numpy().squeeze()

      # get CRS and transform from input file
      with rasterio.open(file_path) as src:
            crs = src.crs
            transform = src.transform

      output_path = f'{output_folder}/prediction_{i}.tif'

      with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=prediction_np.shape[0], 
            width=prediction_np.shape[1],
            count=1, dtype='float32',
            crs=crs, 
            transform=transform
      ) as dst:
            dst.write(prediction_np, 1)
