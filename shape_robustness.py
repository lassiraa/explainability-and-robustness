import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from utils import CocoDistortion


def load_model(model_name: str, path_to_ft_model: str, device: torch.device):
    assert model_name in ['vit_b_32', 'vgg16']

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, 80)

    if model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=False)
        model.heads[0] = nn.Linear(768, 80)
    
    model.load_state_dict(torch.load(path_to_ft_model))
    model.eval()
    model.to(device)
    return model


def measure_shape_robustness(model: nn.Module,
                             coco_loader: DataLoader):
    prob_diffs = np.zeros((len(coco_loader.dataset), 80))
    batch = coco_loader.batch_size

    for i, (inputs, distorted_inputs, labels) in enumerate(coco_loader):
        inputs = inputs.to(device)
        distorted_inputs = distorted_inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        distorted_outputs = model(distorted_inputs)
        prob_diff = (outputs - distorted_outputs) * labels
        prob_diffs[i*batch:(i+1)*batch] = prob_diff.cpu().detach().numpy()
    
    return prob_diffs


if __name__ == '__main__':
    model_name = 'vit_b_32'
    path2data = "/media/lassi/Data/datasets/coco/images/val2017/"
    path2json = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"
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
        transform=val_transform,
        target_transform=None
    )
    coco_loader = DataLoader(
        coco_dset,
        batch_size=16,
        shuffle=False,
        drop_last=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, params_to_update = load_model(
        model_name, device
    )

    prob_diffs = measure_shape_robustness(model, coco_loader)
