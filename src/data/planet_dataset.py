import os
import torch
import rasterio
import numpy as np

from scipy.ndimage import median_filter
from torch.utils.data import Dataset
from PIL import Image

class PlanetDataset(Dataset):
      """
      A dataset class for Planet satellite images, supporting operations like padding, 
      normalization, and data augmentation.

      Attributes:
        data_dir (str): Path to the directory with image and ground truth data.
        pad (bool): Whether to pad images to a fixed size.
        normalization (callable, optional): Function for normalizing image data.
        transforms (callable, optional): Augmentation transforms to be applied to the images.
        img_folder (str): Subdirectory within data_dir containing the images.
        gt_folder (str): Subdirectory within data_dir containing the ground truth images.
        dataset (list of tuples): List where each tuple contains paths to an image 
            and its corresponding ground truth.

      Methods:
        ndvi(img): Calculates the Normalized Difference Vegetation Index (NDVI) for an image.
        savi(img, L=0.5): Calculates the Soil-Adjusted Vegetation Index (SAVI) for an image.
        ndwi(img): Calculates the Normalized Difference Water Index (NDWI) for an image.
        __len__(): Returns the number of image-ground truth pairs in the dataset.
        pad_img(img_tensor, pad_height, pad_width): Pads the image tensor to the specified dimensions.
        __getitem__(idx): Retrieves the image-ground truth pair at the specified index, 
            applying any specified transformations.
      """
    
      def __init__(self, data_dir: str, pad=False, normalization=None, transforms=None, is_fusion=False):
            self.data_dir = data_dir
            self.pad = pad
            self.normalization = normalization
            self.transforms = transforms
            self.is_fusion = is_fusion

            if is_fusion:
                  self.img_folder = os.path.join(data_dir)   
                  goal_dir = os.path.join(data_dir,  '../../gt') 
                  self.gt_folder = os.path.normpath(goal_dir)
            else:
                  self.img_folder = os.path.join(data_dir, 'images')
                  self.gt_folder = os.path.join(data_dir, 'gt')

            # Get a sorted list of image and ground truth file names
            img_filenames = sorted(os.listdir(self.img_folder))
            gt_filenames = sorted(os.listdir(self.gt_folder))

            img_prefix = 'nicfi_'
            gt_prefix = 'nicfi_gt_'

            # pair each input image with its corresponding ground truth
            self.dataset = []
            for img_name in img_filenames:
                  if img_name.startswith(img_prefix):
                        # construct the corresponding ground truth file name
                        gt_name = img_name.replace(img_prefix, gt_prefix)
                  if gt_name in gt_filenames:
                        img_path = os.path.join(self.img_folder, img_name)
                        gt_path = os.path.join(self.gt_folder, gt_name)
                        self.dataset.append((img_path, gt_path))

      @staticmethod
      def ndvi(img):
            red = img[2, :, :]
            nir = img[3, :, :]
            ndvi = (nir - red) / (nir + red + 1e-10)  # epsilon to avoid division by zero

            # convert to tensor and add channel dimension [1, height, width]
            ndvi = torch.from_numpy(ndvi).unsqueeze(0)

            return ndvi

      @staticmethod
      def savi(img, L=0.5):
            red = img[2, :, :]
            nir = img[3, :, :]
            savi = ((nir - red) / (nir + red + L)) * (1 + L)

            return torch.from_numpy(savi).unsqueeze(0)

      @staticmethod
      def ndwi(img):
            green = img[1, :, :]
            nir = img[3, :, :]
            ndwi = (green - nir) / (green + nir + 1e-10)

            return torch.from_numpy(ndwi).unsqueeze(0)

      def __len__(self):
            return len(self.dataset)

      def pad_img(self, img_tensor, pad_height, pad_width):
            if len(img_tensor.shape) == 3:
                  # handle input images
                  channels, height, width = img_tensor.shape
                  padded_img = torch.zeros((channels, height + pad_height, width + pad_width))
                  padded_img[:, :height, :width] = img_tensor

            elif len(img_tensor.shape) == 2:
                  # handle ground truth
                  height, width = img_tensor.shape
                  padded_img = torch.zeros((height + pad_height, width + pad_width))
                  padded_img[:height, :width] = img_tensor

            else:
                  raise ValueError("Unsupported tensor shape for padding")

            return padded_img


      def __getitem__(self, idx):
            img_path, gt_path = self.dataset[idx]

            # read image
            # conversiong to float32 is done to ensure correct normalization
            with rasterio.open(img_path, 'r') as ds:
                  img = ds.read().astype(np.float32)

            # replace NaN values using a median filter
            img = np.nan_to_num(img, nan=np.median(img[~np.isnan(img)]))
            img = median_filter(img, size=3)

            # compute vegetation indices
            ndvi = self.ndvi(img)
            savi = self.savi(img)
            ndwi = self.ndwi(img)

            # apply normalization
            if self.normalization is not None:
                  img = self.normalization(img)

            # convert input image to pytorch tensor
            img_tensor = torch.from_numpy(img).float()

            # Stack NDVI as an additional channel or use it as needed
            img_tensor = torch.cat((img_tensor, ndvi, savi, ndwi), dim=0)

            # read ground truth and convert to tensor
            gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
            gt_tensor = torch.from_numpy(gt).long() # cross-entropy expects Long

            if self.pad is True:
                  target_height = 384  # Adjust
                  target_width = 384

                  pad_height = (target_height - img_tensor.shape[1] % target_height) % target_height
                  pad_width = (target_width - img_tensor.shape[2] % target_width) % target_width

                  img_tensor = self.pad_img(img_tensor, pad_height, pad_width)
                  gt_tensor = self.pad_img(gt_tensor, pad_height, pad_width)
                  gt_tensor = gt_tensor.long() # convert back to Long type

            return img_tensor, gt_tensor