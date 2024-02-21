from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

# import custom modules
from models.lit_model_binary import LitModelBinary
from visualization.get_predictions import get_predictions
from data.planet_dataset import PlanetDataset
from data.normalization_functions import linear_norm_global_percentile
from visualization.plot_segmentation_outputs import plot_segmentation_outputs

# set seed for reproducibility
seed_everything(42, workers=True)

# load testing set
testing_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/testing_data'
normalization = linear_norm_global_percentile
testing_dataset = PlanetDataset(testing_dir,
                                pad=True,
                                normalization=normalization)

# initialize dataloader
batch_size = 16
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4)

# load the checkpoint
model = LitModelBinary.load_from_checkpoint(checkpoint_path='models/checkpoints/planet-singleclass-epoch=42-val_loss=0.03.ckpt')

# set the model for evaluation
model.eval()
model.freeze()

get_predictions(model, test_loader, indices=[18, 69, 112], output_name='planet-binary-predictions')


predictions_file = 'models/predictions/planet-binary-predictions_[18, 69, 112].pth'
plot_segmentation_outputs = plot_segmentation_outputs(predictions_file, 'planet-binary-predictions')