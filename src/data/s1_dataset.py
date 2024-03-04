import os
import torch
import rasterio
import numpy as np

from scipy.ndimage import median_filter
from torch.utils.data import Dataset
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject

class Sentinel1Dataset(Dataset):
    """
    A dataset class for Sentinel-1 satellite images, designed for use in deep learning models. 
    This class supports operations like resampling to match Planet satellite image resolutions, padding, 
    normalization, and data augmentation. It can be used standalone or in fusion with Planet satellite 
    images for tasks like semantic segmentation or classification.

    Attributes:
        data_dir (str): Path to the directory containing Sentinel-1 image and ground truth data.
        pad (bool): Indicates whether to pad images to a fixed size.
        normalization (callable, optional): A function to normalize image data.
        transforms (callable, optional): Augmentation transforms to be applied to the images.
        is_fusion (bool): Specifies if the dataset is being used in fusion with Planet images,
                          enabling resampling to match Planet images' resolution.
        planet_ref_path (str, optional): Directory path to the Planet images, required if is_fusion is True.
        img_folder (str): Subdirectory within data_dir containing Sentinel-1 images.
        gt_folder (str): Subdirectory within data_dir containing the ground truth images.
        dataset (list of tuples): List where each tuple contains paths to an image and its corresponding ground truth.

    Methods:
        __len__(): Returns the number of image-ground truth pairs in the dataset.
        pad_to_target(img_tensor, target_height, target_width): Pads the image tensor to the specified dimensions.
        __getitem__(idx): Retrieves the image-ground truth pair at the specified index, applying any specified transformations
                          and resampling if used in fusion with Planet images.
    """
    def __init__(self, data_dir: str, pad=False, normalization=None, transforms=None, 
                 is_fusion=False, planet_ref_path=None):
        self.data_dir = data_dir
        self.pad = pad
        self.normalization = normalization
        self.transforms = transforms
        self.is_fusion = is_fusion
        self.planet_ref_path = planet_ref_path

        if is_fusion:
            self.img_folder = os.path.join(data_dir)   
            goal_dir = os.path.join(data_dir,  '../../gt') 
            self.gt_folder = os.path.normpath(goal_dir)
            img_prefix = 's1_'
            gt_prefix = 'nicfi_gt_'
        else:
            self.img_folder = os.path.join(data_dir, 'images')
            self.gt_folder = os.path.join(data_dir, 'gt')
            img_prefix = 's1_'
            gt_prefix = 'resampled_nicfi_gt_'

        img_filenames = sorted(os.listdir(self.img_folder))
        gt_filenames = sorted(os.listdir(self.gt_folder))

        self.dataset = []
        for img_name in img_filenames:
            if img_name.startswith(img_prefix):
                # construct corresponding ground truth file name
                gt_name = img_name.replace(img_prefix, gt_prefix)
            if gt_name in gt_filenames:
                img_path = os.path.join(self.img_folder, img_name)
                gt_path = os.path.join(self.gt_folder, gt_name)
                self.dataset.append((img_path, gt_path))

    def __len__(self):
        return len(self.dataset)

    def pad_to_target(self, img_tensor, target_height=192, target_width=192):
        """
        Pads an image tensor to the target height and width with zeros.
        The padding is applied to the bottom and right edges of the image.
        """
        _, height, width = img_tensor.shape

        if self.is_fusion:
            # match Planet images' dimensions
            target_height = 384
            target_width = 384

        # calculate padding
        pad_height = target_height - height if height < target_height else 0
        pad_width = target_width - width if width < target_width else 0

        # apply padding if needed
        if pad_height > 0 or pad_width > 0:
            padding = (0, pad_width, 0, pad_height)  # pad  on the right and bottom
            img_tensor = torch.nn.functional.pad(img_tensor, padding, "constant", 0)

        return img_tensor

    def __getitem__(self, idx):
        img_path, gt_path = self.dataset[idx]
        

        with rasterio.open(img_path, 'r') as src:
            img = src.read().astype(np.float32)  # VV and VH bands

            if self.is_fusion:
                identifier = os.path.basename(img_path).split('_')[1]
                planet_img_path = os.path.join(self.planet_ref_path, f'nicfi_{identifier}')

                with rasterio.open(planet_img_path) as ref:
                    target_transform, target_width, target_height = ref.transform, ref.width, ref.height
                
                # define the destination array for the reprojected data
                dst_img = np.zeros((src.count, target_height, target_width), dtype=np.float32)
                
                # perform resampling to match Planet images' resolution
                reproject(
                    source=img,
                    destination=dst_img,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                img = dst_img

        # replace NaN values using a median filter
        img = np.nan_to_num(img, nan=np.median(img[~np.isnan(img)]))
        img = median_filter(img, size=3)

        if self.normalization is not None:
            img = self.normalization(img)

        img_tensor = torch.from_numpy(img).float()

        gt = np.array(rasterio.open(gt_path).read(1), dtype=np.float32)
        gt_tensor = torch.from_numpy(gt).long()

        if self.pad:
            img_tensor = self.pad_to_target(img_tensor)
            gt_tensor = self.pad_to_target(gt_tensor.unsqueeze(0)).squeeze(0).long()

        return img_tensor, gt_tensor
