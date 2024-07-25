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
from models.lit_model_standalone import LitModelStandalone
from models.lit_model_lf import LitModelLateFusion
from models.lit_model_ef import LitModelEarlyFusion

from data.normalization import s1_norm, planet_norm

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# set seed for reproducibility
seed_everything(42, workers=True)

# read config file
with open('src/train_test_predict/train_config.json') as f:
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
    normalization = planet_norm
elif datasource_dict[config["datasource"]] == Sentinel1Dataset:
    dataset = Sentinel1Dataset
    normalization = s1_norm
elif datasource_dict[config["datasource"]] == FusionDataset:
    dataset = FusionDataset
    planet_norm = planet_norm
    s1_norm = s1_norm


# create training dataset
training_dir = config["training_dir"]

# handle data fusion case
if datasource_dict[config["datasource"]] == FusionDataset:
    print('Initializing fusion dataset')
    training_dataset = dataset(training_dir,
                            planet_normalization=planet_norm,
                            s1_normalization=s1_norm)
else:
    print('Initializing standalone dataset')
    training_dataset = dataset(training_dir,
                            pad=True,
                            normalization=normalization,
                            transforms=True
                            )

# extract validation subset from the training set
total_size = len(training_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size # get 20% of training set as validation set
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
planet_checkpoint_path = 'models/checkpoints/planet_trial10_resnet34-epoch=98-val_f1score=0.79.ckpt'
s1_checkpoint_path = 'models/checkpoints/s1_trial51-epoch=99-val_f1score=0.67.ckpt'

# define model
if config["mode"] == "late_fusion":
    print('Initializing Late Fusion model')
    model = LitModelLateFusion(
        lr=config["learning_rate"],
        threshold=config["threshold"],
        weight_decay=config["weight_decay"],
        pretrained_streams=True,
        s1_checkpoint=s1_checkpoint_path,
        planet_checkpoint=planet_checkpoint_path
    )
elif config["mode"] == "early_fusion":
    print('Initializing Early Fusion model')
    model = LitModelEarlyFusion(
        lr=config["learning_rate"],
        threshold=config["threshold"],
        weight_decay=config["weight_decay"],
        alpha=config["alpha"],
        gamma=config["gamma"]
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
    mode='max',
    save_last=False
)

# initialize profiler
profiler = PyTorchProfiler()

# set up logger
logger = TensorBoardLogger("tb_logs", name=filename_prefix)

# define trainer and start training
trainer = pl.Trainer(max_epochs=config["epochs"],
                     log_every_n_steps=10,
                     accelerator='gpu',
                     devices=1,
                     detect_anomaly=False,
                     strategy=DDPStrategy(find_unused_parameters=True),
                    #  callbacks=[early_stop_callback, checkpoint_callback],
                    callbacks=[checkpoint_callback],
                     logger=logger,
                    )
# print(model)
trainer.fit(model, train_loader, val_loader)

# save model
# torch.save(model.state_dict(), 'models/' + filename_prefix + '.pth')