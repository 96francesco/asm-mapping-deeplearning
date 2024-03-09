import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import json

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, random_split

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

from models.lit_model_binary import LitModelBinary
from models.lit_model_multiclass import LitModelMulticlass
from models.lit_model_fusion import LitModelBinaryLateFusion

# set seed for reproducibility
seed_everything(42, workers=True)

# read config file
with open('src/models/test_config.json') as f:
    config = json.load(f)

mode_dict = {
    "binary": LitModelBinary,
    "multiclass": LitModelMulticlass,
    "fusion": LitModelBinaryLateFusion
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

# create testing dataset
testing_dir = config["testing_dir"]
normalization = normalization_dict[config["normalization"]]
testing_dataset = dataset(testing_dir,
                         pad=True,
                         normalization=normalization)

# load the checkpoint
checkpoint = config["checkpoint"]
model = LitModelMulticlass.load_from_checkpoint(checkpoint_path=checkpoint)

# set the model for evaluation
model.eval()
model.freeze()

# initialize dataloader
batch_size = config["batch_size"]
test_loader = DataLoader(testing_dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         num_workers=4)

# test the model
trainer = pl.Trainer()
trainer.test(model, test_loader)