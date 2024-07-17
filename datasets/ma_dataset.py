import os
import glob
from torch.utils.data import Dataset
from PIL import Image 
import numpy as np
import albumentations.pytorch as PT
import matplotlib.pyplot as plt
class MADataset(Dataset):
    
    def __init__(self, dataset_path, device, affine_transform=None, color_transform=None): 
        self.data_folder = dataset_path
        self.img_paths = sorted(glob.glob(os.path.join(self.data_folder,"train_sat_temp", "*.tiff")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.data_folder, "train_mask_temp","*.tif")))
        if(len(self.img_paths) != len(self.mask_paths)):
            raise Exception("Number of images and masks differs (imgs: {0}, masks: {1})".format(len(self.img_paths), len(self.mask_paths)))
        self.affine_transform = affine_transform
        self.color_transform = color_transform
        self.device = device

    def __len__(self):
        return len(self.img_paths)

    def crop_center(self,img,cropx,cropy):
        try:
            y,x,_ = img.shape
        except:
            y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_paths[index]))
        mask = np.array(Image.open(self.mask_paths[index]))//255
        image = self.crop_center(image, 1024, 1024)
        mask = self.crop_center(mask, 1024, 1024)

        if(self.affine_transform is not None):
            res = self.affine_transform(image=image, mask=mask)
            image, mask = res["image"], res["mask"]
        if(self.color_transform is not None):
            image = self.color_transform(image=image)["image"]
        res = PT.ToTensorV2()(image=image, mask=mask)
        return res["image"].float().to(self.device), res["mask"].to(self.device).int()