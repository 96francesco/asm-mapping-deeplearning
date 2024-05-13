import os

from torch.utils.data import Dataset

# import custom modules
from data.s1_dataset import Sentinel1Dataset
from data.planet_dataset import PlanetDataset


class FusionDataset(Dataset):
      """
      A dataset class for handling the fusion of Sentinel-1 and Planet-NICFI images for semantic segmentation tasks.
      This class is designed to work with a specific directory structure where images from both sources and ground truth
      data are organized under separate subdirectories within a given root directory.
      
      Parameters:
            root_dir (str): The root directory where the data is stored. This directory should contain
                              'training data' and 'testing data' directories, each with 'gt', 'planet', and 's1' subdirectories.
            train (bool): Whether to use the training data. If False, testing data will be used.
            transforms (callable, optional): Optional transform to be applied on a sample.
            planet_normalization (callable, optional): Optional normalization to be applied on Planet images.
            s1_normalization (callable, optional): Optional normalization to be applied on Sentinel-1 images.      
      Returns:
            The processed images from both datasets, and their ground truths.
      """
      def __init__(self, root_dir, is_inference=False, planet_normalization=None, s1_normalization=None):
            self.root_dir = root_dir
            self.is_inference = is_inference

            # handle inference mode
            if is_inference:
                  planet_data_dir = os.path.join(root_dir, 'images/planet')
                  s1_data_dir = os.path.join(root_dir, 'images/s1')
            else:
                  planet_data_dir = os.path.join(root_dir)
                  s1_data_dir = os.path.join(root_dir)

            # initialize Planet and S1 datasets
            self.planet_dataset = PlanetDataset(data_dir=planet_data_dir, normalization=planet_normalization,
                                                pad=True, is_fusion=True, is_inference=is_inference, transforms=True)
            self.s1_dataset = Sentinel1Dataset(data_dir=s1_data_dir, normalization=s1_normalization,
                                                pad=True, is_fusion=True, is_inference=is_inference, transforms=True,
                                                planet_ref_path=planet_data_dir)

      def __len__(self):
            return min(len(self.planet_dataset), len(self.s1_dataset))

      def __getitem__(self, idx):
            planet_data, gt = self.planet_dataset[idx] if not self.is_inference else (self.planet_dataset[idx], None)
            s1_data, _ = self.s1_dataset[idx] if not self.is_inference else (self.s1_dataset[idx], None)

            if not self.is_inference:
                  return planet_data, s1_data, gt
            else:
                  return planet_data, s1_data

