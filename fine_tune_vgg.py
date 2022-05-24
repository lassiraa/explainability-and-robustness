import torchvision.models as models
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pycocotools.coco import COCO


path2data = "/media/lassi/Data/datasets/coco/images/val2017/"
path2json = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"

coco = COCO(path2json)
coco_cats = coco.getCatIds()


def target_transform(targets: list, cats: list = coco_cats) -> np.ndarray:
    khot_cats = np.zeros(80)
    for target in targets:
        khot_cats[cats.index(target['category_id'])] = 1
    return khot_cats


params = dict(
    lr=0.01,
    batch_size=16,
    epochs=16
)

coco_dset = CocoDetection(
    root=path2data,
    annFile=path2json,
    transform=ToTensor(),
    target_transform=target_transform
)
coco_dloader = DataLoader(
    coco_dset,
    batch_size=params['batch_size'],
    shuffle=True,
    drop_last=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)
#  Using feature extraction so only output layer is fine-tuned
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, 80)
model = model.to(device)

#  Getting all parameters that need to be optimized
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.SGD(params_to_update, lr=params['lr'])
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(params['epochs']):
    model.train()
    for inputs, labels in coco_dloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
