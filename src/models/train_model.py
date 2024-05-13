# import libraries
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import json
import gc

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger

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
from models.lit_model_fusion import LitModelLateFusion

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# set seed for reproducibility
seed_everything(42, workers=True)

# read config file
with open('src/models/train_config.json') as f:
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
    print('Initializing Planet normalization functions')
    normalization_dict = {
    "standardization": planet_standardization,
    "percentile": planet_norm_percentile,
    "minmax": planet_norm_minmax,
}
    dataset = PlanetDataset
elif datasource_dict[config["datasource"]] == Sentinel1Dataset:
    print('Initializing Sentinel-1 normalization functions')
    normalization_dict = {
    "standardization": s1_standardization,
    "minmax": s1_norm_minmax,
    "percentile": s1_norm_percentile,
}
    dataset = Sentinel1Dataset
elif datasource_dict[config["datasource"]] == FusionDataset:
    print('Initializing Fusion normalization functions')
    normalization_dict = {
    "planet_minmax": planet_norm_minmax,
    "planet_percentile": planet_norm_percentile,
    "planet_standardization": planet_standardization,
    "s1_standardization": s1_standardization,
    "s1_minmax": s1_norm_minmax,
    "s1_percentile": s1_norm_percentile
}
    dataset = FusionDataset

# create training dataset
training_dir = config["training_dir"]

# handle data fusion case
if datasource_dict[config["datasource"]] == FusionDataset:
    print('Initializing fusion dataset')
    planet_normalization = normalization_dict[config["planet_normalization"]]
    s1_normalization = normalization_dict[config["s1_normalization"]]
    training_dataset = dataset(training_dir,
                            planet_normalization=planet_normalization,
                            s1_normalization=s1_normalization)
else:
    print('Initializing standalone dataset')
    normalization = normalization_dict[config["normalization"]]
    training_dataset = dataset(training_dir,
                            pad=True,
                            normalization=normalization,
                            transforms=True
                            )

# extract validation subset from the training set
total_size = len(training_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size # get 10% of training set as validation set
train_set, val_set = random_split(training_dataset, [train_size, val_size])

# initialize dataloader
batch_size = config["batch_size"]
train_loader = DataLoader(train_set, 
                        batch_size=batch_size,
                        shuffle=True, num_workers=9,
                        persistent_workers=True, 
                        pin_memory=True,
                        prefetch_factor=2)
val_loader = DataLoader(val_set, 
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=9, 
                        persistent_workers=True, 
                        pin_memory=True,
                        prefetch_factor=2)

# pre-trained checkpoints, for Late Fusion mode
planet_checkpoint_path = 'models/checkpoints/planet-binary-optimized-trial43-epoch=59-val_f1score=0.74.ckpt'
s1_checkpoint_path = 'models/checkpoints/planet_trial10_resnet34-epoch=98-val_f1score=0.79.ckpt'

# define model
if config["mode"] == "fusion":
    print('Initializing fusion model')
    model = LitModelLateFusion(
        lr=config["learning_rate"],
        threshold=config["threshold"],
        weight_decay=config["weight_decay"],
        pretrained_streams=True,
        s1_checkpoint=s1_checkpoint_path,
        planet_checkpoint=planet_checkpoint_path
    )
else:
    print('Initializing standalone model')
    model = mode_dict[config["mode"]]
    # define U-Net model
    unet = smp.Unet(
        encoder_name="resnet34",
        decoder_use_batchnorm=True,
        decoder_attention_type='scse',
        encoder_weights=None,
        in_channels=config["in_channels"],
        classes=1, # 1 for binary classification
        activation=None
    )
    model = model(model=unet,
                in_channels=config["in_channels"],
                lr=config["learning_rate"],
                threshold=config["threshold"],
                weight_decay=config["weight_decay"],
                alpha=config["alpha"],
                gamma=config["gamma"]
    )

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

# initialize profiler
profiler = PyTorchProfiler()

# set up logger
logger = TensorBoardLogger("tb_logs", name=filename_prefix)

# define trainer and start training
trainer = pl.Trainer(max_epochs=config["epochs"],
                     log_every_n_steps=10,
                     accelerator='gpu',
                     devices=2,
                     detect_anomaly=False,
                     strategy=DDPStrategy(find_unused_parameters=True),
                    #  callbacks=[early_stop_callback, checkpoint_callback],
                    callbacks=[checkpoint_callback],
                    #  profiler=profiler,
                     logger=logger,
                    #  accumulate_grad_batches=4,
                    #  precision=16,
                    #  gradient_clip_val=0.5,
                    )
# print(model)
trainer.fit(model, train_loader, val_loader)