# import libraries
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
with open('src/models/train_config.json') as f:
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


# create class weights
class_weight = torch.tensor([133822248 / 4196568])

# create training dataset
training_dir = config["training_dir"]
normalization = normalization_dict[config["normalization"]]
training_dataset = dataset(training_dir,
                           pad=True,
                           normalization=normalization)

# extract validation subset from the training set
total_size = len(training_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size # get 20% of training set
train_set, val_set = random_split(training_dataset, [train_size, val_size])

# initialize dataloader
batch_size = config["batch_size"]
train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0)

# define U-Net model
unet = smp.Unet(
    encoder_name="resnet34",
    decoder_use_batchnorm=True,
    decoder_attention_type='scse',
    encoder_weights=None,
    in_channels=config["in_channels"],
    classes=config["number_of_classes"],
    activation=None
)

# define model
model = mode_dict[config["mode"]]
model = model(model=unet,
            loss=config["loss"],
            optimizer=config["optimizer"],
            lr=config["learning_rate"],
            threshold=config["threshold"],
            weight_decay=config["weight_decay"],
            pos_weight=None if config['class_weight'] == "None" else class_weight)

# define filename for the checkpoints       
filename_prefix = config["filename_prefix"]
filename = filename_prefix + "-{epoch:02d}-{val_f1score:.2f}"

# define callbacks
early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=20,
   verbose=True,
   mode='min')

checkpoint_callback = ModelCheckpoint(
    dirpath="models/checkpoints/",
    filename=filename,
    save_top_k=3,
    verbose=False,
    monitor='val_f1score',
    mode='max'
)

# define trainer and start training
trainer = pl.Trainer(max_epochs=config["epochs"],
                     log_every_n_steps=10,
                     accelerator='gpu',
                     devices=2,
                     detect_anomaly=False,
                     strategy=DDPStrategy(find_unused_parameters=True),
                     callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(model, train_loader, val_loader)