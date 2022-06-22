import random

import torchvision.transforms as transforms
import torch

from utils import CocoDistortion


torch.random.manual_seed(50)
random.seed(50)


if __name__ ==  '__main__':
    test_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomRotation(10),
        transforms.CenterCrop(224)
    ])
    path2data = "/media/lassi/Data/datasets/coco/images/val2017/"
    path2json = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"
    path2idjson = "data/image_to_annotation.json"

    distortion_method = 'perpendicular'
    distort_background = 'smooth_transition'

    coco_dset = CocoDistortion(
        root=path2data,
        annFile=path2json,
        imToAnnFile=path2idjson,
        transform=test_transform,
        target_transform=None,
        distort_background=distort_background,
        distortion_method=distortion_method,
        debug_mode=True
    )

    num_ids = len(coco_dset.ids)
    i = 0
    while True:
        image, image_distorted, target = \
            coco_dset.__getitem__(random.randint(0, num_ids))
        image.save(f'./examples/{i}.png')
        image_distorted.save(f'./examples/{i}_{distortion_method}_{distort_background}.png')
        i += 1
        input("Press Enter to get next image...")