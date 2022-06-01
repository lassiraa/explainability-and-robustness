from random import randint 

import torchvision.transforms as transforms

from utils import CocoDistortion


if __name__ ==  '__main__':
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    path2data = "/media/lassi/Data/datasets/coco/images/val2017/"
    path2json = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"
    path2idjson = "data/image_to_annotation.json"
    coco_dset = CocoDistortion(
        root=path2data,
        annFile=path2json,
        imToAnnFile=path2idjson,
        transform=test_transform,
        target_transform=None
    )

    num_ids = len(coco_dset.ids)
    while True:
        image, image_distorted, target = \
            coco_dset.__getitem__(randint(0, num_ids))
        image.show()
        image_distorted.show()
        input("Press Enter to show next image...")