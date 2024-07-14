import torch
import json
import gc

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

# import custom modules
from data.planet_dataset import PlanetDataset
from data.s1_dataset import Sentinel1Dataset
from data.fusion_dataset import FusionDataset

from models.lit_model_standalone import LitModelStandalone
from models.lit_model_lf import LitModelLateFusion
from models.lit_model_ef import LitModelEarlyFusion

from visualization.get_predictions import get_predictions
from visualization.plot_example_segmentation import plot_segmentation_outputs


# set seed for reproducibility
seed_everything(42, workers=True)

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# read config file
with open('src/train_test_predict/test_config.json') as f:
    config = json.load(f)

mode_dict = {
    "standalone": LitModelStandalone,
    "late_fusion": LitModelLateFusion,
    "early_fusion": LitModelEarlyFusion
}

datasource_dict = {
    "planet": PlanetDataset,
    "s1": Sentinel1Dataset,
    "fusion": FusionDataset
}

if datasource_dict[config["datasource"]] == PlanetDataset:
    dataset = PlanetDataset
elif datasource_dict[config["datasource"]] == Sentinel1Dataset:
    dataset = Sentinel1Dataset
elif datasource_dict[config["datasource"]] == FusionDataset:
    dataset = FusionDataset

# create testing dataset
testing_dir = config["testing_dir"]

if datasource_dict[config["datasource"]] == FusionDataset:
    testing_dataset = dataset(testing_dir,
                            planet_normalization=True,
                            s1_normalization=True)
else:
    testing_dataset = dataset(testing_dir,
                            pad=True,
                            normalization=True,
                            transforms=False)

# load the checkpoint
model = mode_dict[config["mode"]]
model = model.load_from_checkpoint(checkpoint_path=config["checkpoint"])
# model = LitModelLateFusion(is_inference=True)
# model.load_state_dict(torch.load('models/lf_model.pth'))

# set the model for evaluation
model.eval()
model.freeze()
# print(model.hparams)

batch_size = config["batch_size"]
test_loader = DataLoader(testing_dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         num_workers=9)

indices = [18, 69, 200]
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