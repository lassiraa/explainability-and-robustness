import time

import torchvision.models as models
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import wandb

from utils import CocoClassification


def get_model_to_fine_tune(model_name: str, device: torch.device):
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
              optimizer: optim.Adam,
              scheduler: ExponentialLR,
              coco_loader_train: DataLoader,
              coco_loader_val: DataLoader,
              params: dict,
              model_name: str,
              device: torch.device,
              checkpoint_dir: str):
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(params['epochs']):
        t0 = time.time()
        tr_loss = []
        val_loss = []

        model.train()
        
        for inputs, labels in coco_loader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())

        model.eval()

        with torch.no_grad():

            for inputs, labels in coco_loader_val:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss.append(loss.item())

        scheduler.step()
        tr_loss = np.mean(tr_loss)
        val_loss = np.mean(val_loss)
        wandb.log({
            'tr_loss': tr_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0]
        })
        wandb.watch(model)
        print(f'Time for epoch {epoch}: {time.time()-t0}')
        print(f'Tr. loss {tr_loss} | Val. loss {val_loss}')

        #  Save model state every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f'{checkpoint_dir}{model_name}_coco_ep{epoch}.pt'
            )
    
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Finetune network')
    parser.add_argument('--images_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/images/',
                        help='path to coco root directory containing image folders')
    parser.add_argument('--ann_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/annotations/',
                        help='path to root directory containing annotations')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.90,
                        help='gamma for exponential lr scheduler')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
    parser.add_argument('--model_name', type=str, default='vit_b_32',
                        help='name of model used for training',
                        choices=['vit_b_32', 'vgg16'])
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/media/lassi/Data/checkpoints/model.pt',
                        help='path to save checkpoint')
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_project', type=str)
    args = parser.parse_args()

    training_params = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gamma=args.gamma,
        model_name=args.model_name
    )
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f'{args.model_name}-lr{args.lr}-gam{args.gamma}'
    )
    wandb.config = training_params

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    coco_dset_train = CocoClassification(
        root=args.images_dir + 'train2017/',
        annFile=args.ann_dir + 'instances_train2017.json',
        transform=train_transform,
        target_transform=None
    )
    coco_loader_train = DataLoader(
        coco_dset_train,
        batch_size=training_params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    coco_dset_val = CocoClassification(
        root=args.images_dir + 'val2017/',
        annFile=args.ann_dir + 'instances_val2017.json',
        transform=val_transform,
        target_transform=None
    )
    coco_loader_val = DataLoader(
        coco_dset_val,
        batch_size=training_params['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, params_to_update = get_model_to_fine_tune(
        args.model_name, device
    )

    optimizer = optim.Adam(params_to_update, lr=training_params['lr'])
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    model = fine_tune(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        coco_loader_train=coco_loader_train,
        coco_loader_val=coco_loader_val,
        params=training_params,
        model_name=args.model_name,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
