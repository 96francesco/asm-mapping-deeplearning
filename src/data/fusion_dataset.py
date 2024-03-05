import os
import numpy as np
import torch
import rasterio

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import median_filter

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
      
      Returns:
            The processed images from both datasets, and their ground truths.
    """
      def __init__(self, root_dir, train=True, transforms=None, planet_normalization=None, s1_normalization=None):
            # get data directories
            s1_data_dir = os.path.join(root_dir, 'training_data' if train else 'testing_data', 'images', 's1')
            planet_data_dir = os.path.join(root_dir, 'training_data' if train else 'testing_data', 'images', 'planet')

            # initialize Sentinel1Dataset and PlanetDatase
            self.s1_dataset = Sentinel1Dataset(data_dir=s1_data_dir, 
                                               pad=True, 
                                               is_fusion=True, 
                                               planet_ref_path=planet_data_dir)
            self.planet_dataset = PlanetDataset(data_dir=planet_data_dir, 
                                                pad=True, 
                                                is_fusion=True)

            # ensure both datasets have the same length and other initialization details here
            assert len(self.s1_dataset) == len(self.planet_dataset), "Datasets must have the same size"

      def __len__(self):
            return len(self.planet_dataset)

      def __getitem__(self, idx):
            # get preprocessed image and ground truth from the Sentinel-1 dataset
            s1_data, gt = self.s1_dataset[idx]

            # get the preprocessed image and ground truth from the Planet dataset
            planet_data, _ = self.planet_dataset[idx]

            return planet_data, s1_data, gt


