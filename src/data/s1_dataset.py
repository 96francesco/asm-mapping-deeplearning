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

    def pad_img(self, img_tensor, pad_height, pad_width):
        channels, height, width = img_tensor.shape
        padded_img = torch.zeros((channels, height + pad_height, width + pad_width))
        padded_img[:, :height, :width] = img_tensor
        return padded_img

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
            target_height = 192
            target_width = 192
            pad_height = (target_height - img_tensor.shape[1] % target_height) % target_height
            pad_width = (target_width - img_tensor.shape[2] % target_width) % target_width
            img_tensor = self.pad_img(img_tensor, pad_height, pad_width)
            gt_tensor = self.pad_img(gt_tensor.unsqueeze(0), pad_height, pad_width).squeeze(0).long()

        return img_tensor, gt_tensor
