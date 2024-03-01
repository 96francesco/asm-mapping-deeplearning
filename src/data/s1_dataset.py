import os
import torch
import rasterio
import numpy as np

from scipy.ndimage import median_filter
from torch.utils.data import Dataset
from PIL import Image

class Sentinel1Dataset(Dataset):
    """
    !!!
    """
    def __init__(self, data_dir: str, pad=False, normalization=None, transforms=None):
        self.data_dir = data_dir
        self.pad = pad
        self.normalization = normalization
        self.transforms = transforms

        self.img_folder = os.path.join(data_dir, 'images')
        self.gt_folder = os.path.join(data_dir, 'gt')

        img_filenames = sorted(os.listdir(self.img_folder))
        gt_filenames = sorted(os.listdir(self.gt_folder))

        img_prefix = 's1_'
        gt_prefix = 'resampled_nicfi_gt_'

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

        # Calculate padding
        pad_height = target_height - height if height < target_height else 0
        pad_width = target_width - width if width < target_width else 0

        # Apply padding if needed
        if pad_height > 0 or pad_width > 0:
            padding = (0, pad_width, 0, pad_height)  # Padding on the right and bottom
            img_tensor = torch.nn.functional.pad(img_tensor, padding, "constant", 0)

        return img_tensor

    def __getitem__(self, idx):
        img_path, gt_path = self.dataset[idx]

        with rasterio.open(img_path, 'r') as ds:
            img = ds.read().astype(np.float32)  # VV and VH bands

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
