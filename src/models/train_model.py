# import libraries
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

# import custom modules
from data.planet_dataset import PlanetDataset
from data.normalization_functions import linear_norm_global_percentile
from models.lit_model_binary import LitModelBinary
from models.lit_model_multiclass import LitModelMulticlass

# set seed for reproducibility
seed_everything(42, workers=True)

### VARIABLES
N = 11 # 11 for multiclass, 1 for binary

# load training set
training_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/planet/multiclass/training_data'
normalization = linear_norm_global_percentile
training_dataset = PlanetDataset(training_dir,
                                 pad=True,
                                 normalization=normalization)

# extract validation subset from the training set
total_size = len(training_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size # get 20% of training set
train_set, val_set = random_split(training_dataset, [train_size, val_size])

# initialize dataloader
batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4)

# define U-Net model
unet = smp.Unet(
    encoder_name="resnet34",
    decoder_use_batchnorm=True,
    decoder_attention_type='scse',
    encoder_weights=None,
    in_channels=7,
    classes=N,
    activation=None
)

# binary_model = LitModelBinary(model=unet,
#                         num_classes=2,
#                         loss='ce',
#                         optimizer='adam')

multiclass_model = LitModelMulticlass(model=unet,
                            num_classes=N,
                            loss='focal',
                            optimizer='adam')

epochs = 100
filename = "planet-multiclass-2-{epoch:02d}-{val_loss:.2f}"

# define callbacks
early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=10,
   verbose=True,
   mode='min')

checkpoint_callback = ModelCheckpoint(
    dirpath="models/checkpoints/",
    filename=filename,
    save_top_k=3,
    verbose=False,
    monitor='val_loss',
    mode='min'
)

# define trainer and start training
trainer = pl.Trainer(max_epochs=epochs,
                     log_every_n_steps=10,
                     accelerator='gpu',
                     devices=2,
                     detect_anomaly=False,
                     strategy=DDPStrategy(find_unused_parameters=True),
                     callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(multiclass_model, train_loader, val_loader)
