import time

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import CocoToKHot


def get_model_to_fine_tune(model_name: str, device: torch.device):
    assert model_name in ['vit_b_32', 'vgg16']

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        #  Using feature extraction so only output layer is fine-tuned
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 80)

    if model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=True)
        #  Using feature extraction so only output layer is fine-tuned
        for param in model.parameters():
            param.requires_grad = False
        model.heads[0] = nn.Linear(768, 80)

    model = model.to(device)
    #  Getting all parameters that need to be optimized
    params_to_update = []
    for _, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    return model, params_to_update


def fine_tune(model: nn.Module,
              optimizer: optim.SGD,
              coco_loader: DataLoader,
              params: dict):
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(params['epochs']):
        t0 = time.time()
        tr_loss = []
        model.train()
        for inputs, labels in coco_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                tr_loss.append(loss.item())
            
        tr_loss = np.mean(tr_loss)
        print(f'Time for epoch {epoch}: {time.time()-t0}')
        print(f'Training loss {tr_loss}')
    
    return model


if __name__ == '__main__':

    path2data = "/media/lassi/Data/datasets/coco/images/val2017/"
    path2json = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"
    model_name = 'vit_b_32'

    training_params = dict(
        lr=0.01,
        batch_size=64,
        epochs=16
    )

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    coco_dset = CocoDetection(
        root=path2data,
        annFile=path2json,
        transform=train_transform,
        target_transform=CocoToKHot(path2json)
    )
    coco_loader = DataLoader(
        coco_dset,
        batch_size=training_params['batch_size'],
        shuffle=True,
        drop_last=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, params_to_update = get_model_to_fine_tune(
        model_name, device
    )

    optimizer = optim.SGD(params_to_update, lr=training_params['lr'])

    model = fine_tune(model, optimizer, coco_loader, training_params)
    model.save(f'{model_name}_coco.pt')
