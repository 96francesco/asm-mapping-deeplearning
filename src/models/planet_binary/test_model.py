import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

# import custom modules
from models.lit_model_binary import LitModelBinary
from data.normalization_functions import linear_norm_global_percentile
from data.planet_dataset import PlanetDataset

# set seed for reproducibility
seed_everything(42, workers=True)

# load testing set
testing_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/testing_data'
normalization = linear_norm_global_percentile
testing_dataset = PlanetDataset(testing_dir,
                                pad=True,
                                normalization=normalization)

# load the checkpoint
model = LitModelBinary.load_from_checkpoint(checkpoint_path='models/checkpoints/planet-singleclass-epoch=42-val_loss=0.03.ckpt')

# set the model for evaluation
model.eval()
model.freeze()

# initialize dataloader
batch_size = 16
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4)

# test the model
trainer = pl.Trainer()
trainer.test(model, test_loader)

