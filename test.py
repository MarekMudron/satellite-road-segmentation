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

def get_dataset(dataset_label, device):
    normalize = get_normalize_transforms()
    if dataset_label == "deepglobe":
        dataset = DeepGlobeDataset(
            os.path.join(os.getcwd(), "data", "train"),
            device=device,
            color_transform=normalize)
    elif dataset_label == "ma":
        dataset = MADataset(
            os.path.join(os.getcwd(), "data"),
            device=device,
            color_transform=normalize)
    elif dataset_label == "both":
        dataset = CombinedDataset(
        os.path.join(os.getcwd(), "data", "train"),
        os.path.join(os.getcwd(), "data"),
        device=device,
        color_transform=normalize)
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

def get_normalize_transforms():
    # affine transforms are applied both to image and mask
    # color transforms apply to image only
    return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def eval_model(model, test_set):
    
    sum_iou = 0
    model.eval()
    criterion = smp.losses.JaccardLoss(mode="binary")
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
    for img, mask in tqdm(test_dataloader, desc="evaluating"):
        pred = model(img)
        iou = criterion(pred, mask)
        sum_iou += iou.item()
    len_dl = len(test_dataloader)
    avg_iou = sum_iou / len_dl
    return 1 - avg_iou

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_channels = (256, 128, 64, 32, 16)

    encoder_name, encoder_depth, trained_on, best_model_path = parse_path(args.model_path)
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

    print("Evaluating ...\nEncoder: {0}\nencoder-depth: {1}\ntrained on {2}\ntesting on {3}\nmodel path{4}".format( 
          encoder_name,
          encoder_depth,
          trained_on,
          args.data,
          args.model_path))
    iou = eval_model(model, test_set)
    print("IOU: ", iou)
    print("="*60)

if __name__ == '__main__':
    main()