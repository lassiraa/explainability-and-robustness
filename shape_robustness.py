import json
from cv2 import correctMatches, threshold

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn import metrics

from utils import CocoDistortion


def load_model(
    model_name: str,
    device: torch.device
) -> nn.Module:
    if 'vgg' in model_name:
        model = getattr(models, model_name)(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 80)

    if 'vit_' in model_name:
        model = getattr(models, model_name)(pretrained=False)
        #  Required as vit_l models have different in_features than vit_b models
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, 80)

    if 'resnet' in model_name:
        model = getattr(models, model_name)(pretrained=False)
        model.fc = nn.Linear(2048, 80)
    
    model.load_state_dict(torch.load(f'{model_name}_coco.pt'))
    model.eval()
    model.to(device)
    return model


def measure_shape_robustness(
    model: nn.Module,
    coco_loader: DataLoader,
    device: torch.device
) -> tuple[np.ndarray]:
    preds = np.zeros((len(coco_loader.dataset), 80))
    preds_dist = np.zeros((len(coco_loader.dataset), 80))
    distorted_classes = np.zeros((len(coco_loader.dataset), 80))
    all_classes = np.zeros((len(coco_loader.dataset), 80))
    batch = coco_loader.batch_size
    activation = nn.Sigmoid()

    with torch.no_grad():

        for i, (inputs, distorted_inputs, distorted_label, all_labels) in enumerate(coco_loader):
            inputs = inputs.to(device)
            distorted_inputs = distorted_inputs.to(device)
            outputs = activation(model(inputs))
            distorted_outputs = activation(model(distorted_inputs))
            preds[i*batch:(i+1)*batch] = outputs.cpu().detach().numpy()
            preds_dist[i*batch:(i+1)*batch] = distorted_outputs.cpu().detach().numpy()
            distorted_classes[i*batch:(i+1)*batch] = distorted_label.numpy()
            all_classes[i*batch:(i+1)*batch] = all_labels.numpy()
    
    return preds, preds_dist, distorted_classes, all_classes


def calculate_statistics(
    preds: np.ndarray,
    preds_dist: np.ndarray,
    distorted_classes: np.ndarray,
    all_classes: np.ndarray
) -> tuple[float]:
    #  Find optimal threshold by F1-score with margin of 0.05
    threshold = 0.05
    best_score = (0, 0)
    while threshold < 1:
        f1_score = metrics.f1_score(all_classes, np.where(preds > threshold, 1, 0))
        threshold += 0.05
        if f1_score > best_score[0]:
            best_score = (f1_score, threshold)
    threshold = best_score[1]

    #  Check how binary predictions change with distortions
    preds *= distorted_classes
    preds_dist *= distorted_classes
    class_preds = preds.max(axis=1)
    class_preds_dist = preds_dist.max(axis=1)
    distort_ratio = len(class_preds[class_preds > threshold]) \
         / len(class_preds_dist[class_preds_dist > threshold])
    distort_ratio_strict = len(class_preds[class_preds > threshold + 0.05]) \
         / len(class_preds_dist[class_preds_dist > threshold + 0.05])
    distort_ratio_lenient = len(class_preds[class_preds > threshold - 0.05]) \
         / len(class_preds_dist[class_preds_dist > threshold - 0.05])
    accuracy = len(class_preds[class_preds > threshold])
    return accuracy, distort_ratio, distort_ratio_strict, distort_ratio_lenient


def get_model_robustness(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    distortion_method: str,
    distort_background: str,
    path2data: str,
    path2json: str,
    path2idjson: str,
    batch_size: int,
    num_workers: int
) -> None:
    if 'vit_' in model_name:
        mean = [.5, .5, .5]
        std = [.5, .5, .5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

    coco_dset = CocoDistortion(
        root=path2data,
        annFile=path2json,
        imToAnnFile=path2idjson,
        transform=val_transform,
        target_transform=None,
        distort_background=distort_background,
        distortion_method=distortion_method
    )
    coco_loader = DataLoader(
        coco_dset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    preds, preds_dist, distorted_classes, all_classes = measure_shape_robustness(model, coco_loader, device)

    return calculate_statistics(preds, preds_dist, distorted_classes, all_classes)


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

    model = load_model(args.model_name, device)

    distortion_methods = ['perpendicular', 'random_noise']
    background_distortion_methods = ['blur', None]
    results = []

    for distortion_method in distortion_methods:
        
        for distort_background in background_distortion_methods:
            print('Processing', distortion_method, distort_background)
            
            accuracy, distort_ratio, distort_ratio_strict, distort_ratio_lenient \
                = get_model_robustness(
                model=model,
                model_name=args.model_name,
                device=device,
                distortion_method=distortion_method,
                distort_background=distort_background,
                path2data=args.images_dir,
                path2json=args.ann_path,
                path2idjson=args.id_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )

            results.append(dict(
                model_name=args.model_name,
                distortion_method=distortion_method,
                distort_background=distort_background,
                accuracy=accuracy,
                distort_ratio=distort_ratio,
                distort_ratio_strict=distort_ratio_strict,
                distort_ratio_lenient=distort_ratio_lenient
            ))

    #  Save image to annotation dictionary as json
    with open(f'data/{args.model_name}_shape_robustness.json', 'w') as fp:
        json.dump(results, fp)
