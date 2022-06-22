import json
from typing import Any

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import CocoExplainabilityMeasurement, calculate_mass_within


#  TODO: move to utils
def load_model(
    model_name: str,
    device: torch.device
) -> nn.Module:
    if 'vgg' in model_name:
        model = getattr(models, model_name)(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 80)
        target_layers = [model.features[-1]]

    if 'vit_' in model_name:
        model = getattr(models, model_name)(pretrained=False)
        #  Required as vit_l models have different in_features than vit_b models
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, 80)
        target_layers = [model.encoder.layers[-1].ln_1]

    if 'resnet' in model_name:
        model = getattr(models, model_name)(pretrained=False)
        model.fc = nn.Linear(2048, 80)
        target_layers = [model.layer4[-1]]
    
    model.load_state_dict(torch.load(f'{model_name}_coco.pt'))
    model.eval()
    model.to(device)
    return model, target_layers


def measure_weighting_game(
    model: nn.Module,
    coco_loader: DataLoader,
    device: torch.device,
    target_layers: Any
) -> tuple[np.ndarray]:
    preds = np.zeros((len(coco_loader.dataset), 80))
    preds_dist = np.zeros((len(coco_loader.dataset), 80))
    classes = np.zeros((len(coco_loader.dataset), 80))
    batch = coco_loader.batch_size


    cam = GradCAM(model, target_layers, use_cuda=True)

    with torch.no_grad():

        for i, (inputs, class_to_targets) in enumerate(coco_loader):
            inputs = inputs.to(device)
            for idx, target in class_to_targets.items():
                mask = target['mask'].to(device)
                saliency_map = cam(inputs, [ClassifierOutputTarget(idx)])
                print(calculate_mass_within(saliency_map, mask))
    
    return preds, preds_dist, classes


def calculate_statistics(
    preds: np.ndarray,
    preds_dist: np.ndarray,
    classes: np.ndarray
) -> tuple[float]:
    preds *= classes
    preds_dist *= classes
    class_preds = preds.max(axis=1)
    class_preds_dist = preds_dist.max(axis=1)
    mean = np.mean(class_preds)
    mean_distort = np.mean(class_preds_dist)
    distort_ratio = np.mean(class_preds_dist) / np.mean(class_preds)
    return mean, mean_distort, distort_ratio


def get_explanation_robustness(
    model: nn.Module,
    device: torch.device,
    target_layers: Any,
    path2data: str,
    path2json: str,
    batch_size: int,
    num_workers: int
) -> None:
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    coco_dset = CocoExplainabilityMeasurement(
        root=path2data,
        annFile=path2json,
        transform=image_transform,
        target_transform=mask_transform
    )
    coco_loader = DataLoader(
        coco_dset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    preds, preds_dist, classes = measure_weighting_game(model, coco_loader, device, target_layers)

    return calculate_statistics(preds, preds_dist, classes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Measure shape robustness')
    parser.add_argument('--images_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/images/val2017/',
                        help='path to coco root directory containing image folders')
    parser.add_argument('--ann_path', type=str,
                        default='/media/lassi/Data/datasets/coco/annotations/instances_val2017.json',
                        help='path to root directory containing annotations')
    parser.add_argument('--id_path', type=str,
                        default='data/image_to_annotation.json',
                        help='path to root directory containing annotations')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
    parser.add_argument('--model_name', type=str, default='vit_b_32',
                        help='name of model used for inference',
                        choices=[
                            'vit_b_32', 'vit_b_16', 'vit_l_32', 'vit_l_16',
                            'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn',
                            'resnet50', 'resnet101', 'resnet152'
                        ])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, target_layers = load_model(args.model_name, device)

            
    mean, mean_distort, distort_ratio = get_explanation_robustness(
        model=model,
        device=device,
        target_layers=target_layers,
        path2data=args.images_dir,
        path2json=args.ann_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )