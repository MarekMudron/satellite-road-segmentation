import argparse
import os
import segmentation_models_pytorch as smp
from datasets.deep_globe_dataset import DeepGlobeDataset
from datasets.ma_dataset import MADataset
from datasets.combined_dataset import CombinedDataset
import torch
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.early_stopper import EarlyStopper
import datetime

import numpy as np
RANDOM_SEED=66
torch.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)
START_TIME = str(datetime.datetime.now().time())
LR = 0.0003

def parseargs():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('-e', '--encoder', choices=[
        "resnet18", "mobilenet_v2", "vgg11"
    ], default="resnet18")
    parser.add_argument('--encoder-depth', type=int, choices=range(3,6), default=5)
    parser.add_argument('--pretrained-encoder', action=argparse.BooleanOptionalAction)   

    parser.add_argument('--train-data', required=True, choices=[
        "deepglobe", "ma", "both"
    ])
    parser.add_argument('--augment-data', action=argparse.BooleanOptionalAction)   
    args = parser.parse_args()
    return args

def get_transforms(should_augment):
    # affine transforms are applied both to image and mask
    # color transforms apply to image only
    if should_augment:
        affine_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
        color_transform = A.Compose([
            A.GaussianBlur(blur_limit=(5,5)),
            A.HueSaturationValue(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # if no augmenation is desired, we only normalize input image cuz its gooood
        affine_transform = None
        color_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return affine_transform, color_transform

def get_dataset(dataset_label, should_augment, device):
    affine_transform, color_transform = get_transforms(should_augment)

    if dataset_label == "deepglobe":
        dataset = DeepGlobeDataset(
            os.path.join(os.getcwd(), "data", "train"),
            device=device,
            affine_transform=affine_transform, 
            color_transform=color_transform)
    elif dataset_label == "ma":
        dataset = MADataset(
            os.path.join(os.getcwd(), "data"),
            device=device,
            affine_transform=affine_transform, 
            color_transform=color_transform)
    elif dataset_label == "both":
        dataset = CombinedDataset(
        os.path.join(os.getcwd(), "data", "train"),
        os.path.join(os.getcwd(), "data"),
        device=device,
        affine_transform=affine_transform, 
        color_transform=color_transform)
    return dataset

def eval_model(model, criterion, val_dataloader):
    sum_iou = 0
    model.eval()
    for img, mask in tqdm(val_dataloader, desc="evaluating"):
        pred = model(img)
        iou = criterion(pred, mask)
        sum_iou += iou.item()
    len_dl = len(val_dataloader)
    avg_iou = sum_iou / len_dl
    model.train()
    return avg_iou

def save_model(model, dir, filename):
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(),os.path.join(dir, filename) )

def train(model, train_set, val_set, writer):
    
    train_dataloader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=2, shuffle=True)

    model.train()
    criterion = smp.losses.JaccardLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    if(isinstance(train_set.dataset, MADataset)):
        LOG_EVERY_X_BATCH = 50
    else:
        LOG_EVERY_X_BATCH = 500
    
    MAX_EPOCHS = 5
    step = 0
    early_stopper = EarlyStopper(patience=3)
    quit_training = False
    for epoch in range(MAX_EPOCHS):
        print("EPOCH {0}/{1}".format(epoch, MAX_EPOCHS))
        for batch, (img, mask) in enumerate(tqdm(train_dataloader,desc="training")):
            # Compute prediction and loss
            pred = model(img)
            loss = criterion(pred, mask)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()
            writer.add_scalar("IoU/train", loss, step)
            if (batch % LOG_EVERY_X_BATCH == 0) or (batch == len(train_dataloader)-1):
                val_iou = eval_model(model, criterion, val_dataloader)
                writer.add_scalar("IoU/test", val_iou, step)
                save_model(model, os.path.join("models", writer.get_logdir() ), "epoch({:04d})-batch({:04d})".format(epoch, batch))
                #if epoch > 0:# and early_stopper.early_stop(val_iou):
                    #quit_training = True
                    #break
            step+=1
        if quit_training:           
            break
    
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

def args_to_string(args):
    l = "{0}-{1}-{2}".format(
        args.encoder, 
        args.encoder_depth,
        args.train_data)
    if(args.augment_data):
        l += "-augmented"
    if(args.pretrained_encoder):
        l += "-pretrained"
    return l


def main():
    args = parseargs()
    encoder = args.encoder
    encoder_depth = args.encoder_depth
    use_pretrained = args.pretrained_encoder
    decoder_channels = (256, 128, 64, 32, 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights="imagenet" if use_pretrained else None, 
        in_channels=3, 
        decoder_channels=decoder_channels[:encoder_depth],
        encoder_depth=encoder_depth,
        classes=1)
    
    model.to(device)

    dataset = get_dataset(args.train_data, args.augment_data, device)
    train_set, val_set, _ = split_data(dataset)
    train(model, train_set, val_set, SummaryWriter(os.path.join("runs", "train", args_to_string(args)+ START_TIME)))

if __name__ == '__main__':
    main()