import os
import glob
from torch.utils.data import Dataset
from PIL import Image 
import numpy as np
import albumentations.pytorch as PT
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from datasets.ma_dataset import MADataset
from datasets.deep_globe_dataset import DeepGlobeDataset

class CombinedDataset(ConcatDataset):
    
    def __init__(self, deepglobe_path, ma_path, device,affine_transform=None, color_transform=None):
        self.ma = MADataset(
            ma_path,
            device=device,
            affine_transform=affine_transform, 
            color_transform=color_transform)
        self.deepglobe = DeepGlobeDataset(
            deepglobe_path, 
            device=device,
            affine_transform=affine_transform, 
            color_transform=color_transform
        )
        super(CombinedDataset, self).__init__([self.deepglobe, self.ma])