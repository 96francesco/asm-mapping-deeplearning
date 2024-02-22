from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

# import custom modules
from models.lit_model_binary import LitModelBinary
from models.lit_model_multiclass import LitModelMulticlass
from visualization.get_predictions import get_predictions
from data.planet_dataset import PlanetDataset
from data.normalization_functions import linear_norm_global_percentile
from visualization.plot_binary_segmentation import plot_segmentation_outputs as plt_binary
from visualization.plot_multiclass_segmentation import plot_segmentation_outputs as plt_multiclass

# set seed for reproducibility
seed_everything(42, workers=True)

# load testing set
testing_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/planet/multiclass/testing_data'
normalization = linear_norm_global_percentile
testing_dataset = PlanetDataset(testing_dir,
                                pad=True,
                                normalization=normalization)

# initialize dataloader
batch_size = 16
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4)

# load the checkpoint
# model = LitModelBinary.load_from_checkpoint(checkpoint_path='models/checkpoints/planet-singleclass-epoch=42-val_loss=0.03.ckpt')
model = LitModelMulticlass.load_from_checkpoint(checkpoint_path='models/checkpoints/planet-multiclass-2-epoch=32-val_loss=0.02.ckpt')

# set the model for evaluation
model.eval()
model.freeze()

# get_predictions(model, test_loader, indices=[18, 69, 112], output_name='planet-binary-predictions')
get_predictions(model, test_loader, indices=[13, 222, 100], output_name='planet-multiclass-2-predictions')

# predictions_file = 'models/predictions/planet-binary-predictions_[18, 69, 112].pth'
# plot_segmentation_outputs = plt_binary(predictions_file, 'planet-binary-predictions')

predictions_file = 'models/predictions/planet-multiclass-2-predictions_[13, 222, 100].pth'
plot_segmentation_outputs = plt_multiclass(predictions_file, 'planet-multiclass-2-predictions')