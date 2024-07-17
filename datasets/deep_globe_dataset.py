import os
import glob
from torch.utils.data import Dataset
from PIL import Image 
import numpy as np
import albumentations.pytorch as PT

class DeepGlobeDataset(Dataset):
    
    def __init__(self, dataset_path, device, affine_transform=None, color_transform=None): 
        self.data_folder = dataset_path
        self.img_paths = sorted(glob.glob(os.path.join(self.data_folder, "*_sat.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.data_folder, "*_mask.png")))

        if(len(self.img_paths) != len(self.mask_paths)):
            raise Exception("Number of images and masks differs (imgs: {0}, masks: {1})".format(len(self.img_paths), len(self.mask_paths)))
        self.affine_transform = affine_transform
        self.color_transform = color_transform
        self.device = device

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_paths[index]))
        mask = np.array(Image.open(self.mask_paths[index]))

        if(len(mask.shape) == 3):
            mask = mask[:,:,0]
        elif(len(mask.shape) == 2):
            unique, counts = np.unique(mask, return_counts=True)
            mask = mask
        mask = mask//255
        if(self.affine_transform is not None):
            res = self.affine_transform(image=image, mask=mask)
            image, mask = res["image"], res["mask"]
        if(self.color_transform is not None):
            image = self.color_transform(image=image)["image"]
        res = PT.ToTensorV2()(image=image, mask=mask)
        return res["image"].float().to(self.device), res["mask"].to(self.device)