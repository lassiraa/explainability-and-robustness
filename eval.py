import time

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from utils import CocoClassification


def load_model(
    model_name: str,
    device: torch.device
) -> nn.Module:
    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 80)

    if model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, 80)

    if model_name == 'swin_t':
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 80)

    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 80)
    
    model.load_state_dict(torch.load(f'{model_name}_coco.pt'))
    model.eval()
    model.to(device)
    return model


def evaluate_model(
    model: nn.Module,
    coco_loader_val: DataLoader,
    device: torch.device
    ) -> None:

    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    val_loss = []

    model.eval()

    with torch.no_grad():

        for inputs, labels in coco_loader_val:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss.append(loss.item())

    val_loss = np.mean(val_loss)
    print(f'Time for evaluation: {time.time()-t0}')
    print(f'Val. loss {val_loss}')
    
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--images_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/images/',
                        help='path to coco root directory containing image folders')
    parser.add_argument('--ann_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/annotations/',
                        help='path to root directory containing annotations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
    parser.add_argument('--model_name', type=str, default='vit_b_32',
                        help='name of model',
                        choices=['vit_b_32', 'vgg16_bn', 'resnet50', 'swin_t'])
    args = parser.parse_args()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

    coco_dset_val = CocoClassification(
        root=args.images_dir + 'val2017/',
        annFile=args.ann_dir + 'instances_val2017.json',
        transform=val_transform,
        target_transform=None
    )
    coco_loader_val = DataLoader(
        coco_dset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_name, device)

    model = evaluate_model(
        model=model,
        coco_loader_val=coco_loader_val,
        device=device
    )
