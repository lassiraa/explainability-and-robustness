import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from utils import CocoDistortion


def load_model(model_name: str, device: torch.device):
    assert model_name in ['vit_b_32', 'vgg16']

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 80)

    if model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=False)
        model.heads[0] = nn.Linear(768, 80)
    
    model.load_state_dict(torch.load(f'{model_name}_coco.pt'))
    model.eval()
    model.to(device)
    return model


def measure_shape_robustness(model: nn.Module,
                             coco_loader: DataLoader):
    preds = np.zeros((len(coco_loader.dataset), 80))
    preds_dist = np.zeros((len(coco_loader.dataset), 80))
    classes = np.zeros((len(coco_loader.dataset), 80))
    batch = coco_loader.batch_size
    activation = nn.Sigmoid()

    with torch.no_grad():

        for i, (inputs, distorted_inputs, labels) in enumerate(coco_loader):
            inputs = inputs.to(device)
            distorted_inputs = distorted_inputs.to(device)
            labels = labels.to(device)
            outputs = activation(model(inputs))
            distorted_outputs = activation(model(distorted_inputs))
            preds[i*batch:(i+1)*batch] = outputs.cpu().detach().numpy()
            preds_dist[i*batch:(i+1)*batch] = distorted_outputs.cpu().detach().numpy()
            classes[i*batch:(i+1)*batch] = labels.cpu().detach().numpy()
    
    return preds, preds_dist, classes


if __name__ == '__main__':
    model_name = 'vgg16'
    path2data = '/media/lassi/Data/datasets/coco/images/val2017/'
    path2json = '/media/lassi/Data/datasets/coco/annotations/instances_val2017.json'
    path2idjson = 'data/image_to_annotation.json'
    path2model = f'{model_name}_coco.pt'

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    coco_dset = CocoDistortion(
        root=path2data,
        annFile=path2json,
        imToAnnFile=path2idjson,
        transform=val_transform,
        target_transform=None
    )
    coco_loader = DataLoader(
        coco_dset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=16
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(model_name, device)

    preds, preds_dist, classes = measure_shape_robustness(model, coco_loader)
    np.save(f'data/{model_name}_preds.npy', preds)
    np.save(f'data/{model_name}_preds_dist.npy', preds_dist)
    np.save(f'data/{model_name}_classes.npy', classes)
