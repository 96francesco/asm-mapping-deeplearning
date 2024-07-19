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

from utils.visualization.get_probabilities import get_probabilities
from utils.visualization.plot_example_segmentation import plot_segmentation_outputs


# set seed for reproducibility
seed_everything(42, workers=True)

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# read config file
with open('src/train_test_predict/predict_config.json') as f:
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
# get probabilites
probs = get_probabilities(model, 
                        test_loader, 
                        mode=config["mode"],
                        indices=config['indices'], 
                        output_name=config["output_name"])

# plot examples
plot_segmentation_outputs = plot_segmentation_outputs(probs, 
                                                      mode=config["mode"],
                                                      data_source=config["datasource"],
                                                      threshold=config["threshold"],
                                                      original_shape=config["original_shape"],
                                                      output_name=config["output_name"])
print("Predictions saved successfully!")