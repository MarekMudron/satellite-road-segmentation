import argparse
import os
import segmentation_models_pytorch as smp
from datasets.deep_globe_dataset import DeepGlobeDataset
from datasets.ma_dataset import MADataset
from datasets.combined_dataset import CombinedDataset
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

RANDOM_SEED=66
torch.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)


def parseargs():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('-m', '--model-path',required=True)

    parser.add_argument('--data', required=True, choices=[
        "deepglobe", "ma", "both"
    ])
    args = parser.parse_args()
    return args

def get_normalize_transforms():
    # affine transforms are applied both to image and mask
    # color transforms apply to image only
    return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_unnormalize_transforms():   
    return A.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

def get_dataset(dataset_label, device):
    normalize_transforms = get_normalize_transforms()
    if dataset_label == "deepglobe":
        dataset = DeepGlobeDataset(
            os.path.join(os.getcwd(), "data", "train"),
            device=device,
            color_transform=normalize_transforms)
    elif dataset_label == "ma":
        dataset = MADataset(
            os.path.join(os.getcwd(), "data"),
            device=device,
            color_transform=normalize_transforms)
    elif dataset_label == "both":
        dataset = CombinedDataset(
        os.path.join(os.getcwd(), "data", "train"),
        os.path.join(os.getcwd(), "data"),
        device=device,
        color_transform=normalize_transforms)
    return dataset

def split_data(dataset):
    from torch.utils.data import random_split

    validation_split = .10
    test_split = .15
    train_split = 1 - validation_split - test_split


    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train, val, test = random_split(dataset, [
        train_split,
        validation_split,
        test_split
    ], generator=generator)
    return train, val, test

def iou(pred, mask):
    pred = pred.squeeze().squeeze().detach().numpy()
    pred = (pred > 0.5).astype(int)
    mask = mask.squeeze().squeeze().detach().numpy()
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection

    # Calculate IoU
    smooth = 1e-6
    iou = (intersection + smooth) / (union + smooth)
    return iou

def visualize(model_name, model, test_set):
    model.eval()
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

    unnormalize = get_unnormalize_transforms()
    i = 0
    for img, mask in test_dataloader:
        pred = model(img)
        pred_sigmoided = pred.sigmoid()
        fig, ax = plt.subplots(1,3, figsize=(16,6))
        ax[0].imshow((unnormalize(image=img[0].permute(1,2,0).detach().numpy()*255)["image"]*255).astype(int))
        ax[0].set_title("Input image")
        ax[1].imshow(mask[0].detach().numpy())
        ax[1].set_title("Ground truth")
        ax[2].imshow(pred.sigmoid()[0][0].detach().numpy())
        ax[2].set_title("Prediction, IoU: {:.2f}".format(iou(pred_sigmoided, mask)))
        
        plt.savefig(f"{model_name}_{i:02d}.png")
        plt.show()
        i+=1
        

def parse_path(model_path):
    model_fn = os.path.basename(model_path)
    p = model_fn.split("-")
    encoder_name = p[0]
    encoder_depth = p[1]
    best_model_path = sorted(os.listdir(model_path))[-1]
    trained_on = p[2]
    return encoder_name, int(encoder_depth),  trained_on, os.path.join(model_path, best_model_path)

def main():
    args = parseargs()
    device = 'cpu'
    decoder_channels = (256, 128, 64, 32, 16)

    encoder_name, encoder_depth, _, best_model_path = parse_path(args.model_path)
    model = smp.Unet(
        encoder_name=encoder_name, 
        encoder_weights=None, 
        in_channels=3, 
        decoder_channels=decoder_channels[:encoder_depth],
        encoder_depth=encoder_depth,
        classes=1)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model.to(device)

    dataset = get_dataset(args.data, device)
    _, _, test_set = split_data(dataset)

    visualize(os.path.basename(args.model_path), model, test_set)

if __name__ == '__main__':
    main()