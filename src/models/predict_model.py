import torch
import json
import gc

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

# import custom modules
from data.planet_dataset import PlanetDataset
from data.s1_dataset import Sentinel1Dataset
from data.fusion_dataset import FusionDataset

from data.planet_dataset_normalization import linear_norm_global_percentile as planet_norm_percentile
from data.planet_dataset_normalization import linear_norm_global_minmax as planet_norm_minmax
from data.planet_dataset_normalization import global_standardization as planet_standardization

from data.s1_dataset_normalization import global_standardization as s1_standardization
from data.s1_dataset_normalization import linear_norm_global_minmax as s1_norm_minmax
from data.s1_dataset_normalization import linear_norm_global_percentile as s1_norm_percentile

from models.lit_model_standalone import LitModelStandalone
from models.lit_model_lf import LitModelLateFusion

from visualization.get_predictions import get_predictions
from visualization.plot_example_segmentation import plot_segmentation_outputs


# set seed for reproducibility
seed_everything(42, workers=True)

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# read config file
with open('src/models/test_config.json') as f:
    config = json.load(f)

mode_dict = {
    "standalone": LitModelStandalone,
    "fusion": LitModelLateFusion
}

datasource_dict = {
    "planet": PlanetDataset,
    "s1": Sentinel1Dataset,
    "fusion": FusionDataset
}

if datasource_dict[config["datasource"]] == PlanetDataset:
    normalization_dict = {
    "standardization": planet_standardization,
    "percentile": planet_norm_percentile,
    "minmax": planet_norm_minmax,
}
    dataset = PlanetDataset
elif datasource_dict[config["datasource"]] == Sentinel1Dataset:
    normalization_dict = {
    "standardization": s1_standardization,
    "minmax": s1_norm_minmax,
    "percentile": s1_norm_percentile,
}
    dataset = Sentinel1Dataset
elif datasource_dict[config["datasource"]] == FusionDataset:
    normalization_dict = {
    "planet_minmax": planet_norm_minmax,
    "planet_percentile": planet_norm_percentile,
    "planet_standardization": planet_standardization,
    "s1_standardization": s1_standardization,
    "s1_minmax": s1_norm_minmax,
    "s1_percentile": s1_norm_percentile
}
    dataset = FusionDataset

# create testing dataset
testing_dir = config["testing_dir"]

if datasource_dict[config["datasource"]] == FusionDataset:
    planet_normalization = normalization_dict[config["planet_normalization"]]
    s1_normalization = normalization_dict[config["s1_normalization"]]
    testing_dataset = dataset(testing_dir,
                            planet_normalization=planet_normalization,
                            s1_normalization=s1_normalization)
else:
    normalization = normalization_dict[config["normalization"]]
    testing_dataset = dataset(testing_dir,
                            pad=True,
                            normalization=normalization,
                            transforms=False)

# load the checkpoint
model = mode_dict[config["mode"]]
model = model.load_from_checkpoint(checkpoint_path=config["checkpoint"])

# set the model for evaluation
model.eval()
model.freeze()
print(model.hparams)

batch_size = config["batch_size"]
test_loader = DataLoader(testing_dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         num_workers=9)


indices = [0, 50, 170]
filename = config['checkpoint_name']
print(filename)
get_predictions(model, 
                test_loader, 
                is_fusion=False,
                indices=indices, 
                output_name=f'{filename}')

predictions_file = f'models/predictions/{filename}-{indices}.pth'
plot_segmentation_outputs = plot_segmentation_outputs(predictions_file, 
                                                      f'{filename}_{indices}',
                                                      is_fusion=False,
                                                      is_optical=True,
                                                      threshold=0.4,
                                                      original_dimensions=(375, 375))