from typing import Any, Callable, Optional, Tuple, List
import os
import json

import numpy as np
import cv2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from PIL import Image


rng = np.random.default_rng(51)


class CocoClassification(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        categories = self.coco.getCatIds()
        self.num_categories = len(categories)
        self.categories = {cat: idx for idx, cat in enumerate(categories)}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        labels = np.zeros(self.num_categories, dtype=np.float32)

        for ann in anns:
            assert('category_id' in ann)
            idx = self.categories[ann['category_id']]
            labels[idx] = 1
        
        return labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoDistortion(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        imToAnnFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        with open(imToAnnFile, 'r') as rf:
            self.im_to_ann = json.load(rf)
            self.im_to_ann = {
                int(img_id): ann_id
                for img_id, ann_id in self.im_to_ann.items()
            }
        self.ids = list(sorted(self.im_to_ann.keys()))
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        categories = self.coco.getCatIds()
        self.num_categories = len(categories)
        self.categories = {cat: idx for idx, cat in enumerate(categories)}

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.im_to_ann[id])[0]
    
    def _warp_image(
        self,
        image: Image.Image,
        segmentation: np.ndarray,
        area: float
    ) -> Image.Image:
        target = segmentation.copy()
        noise_strength = max(min(np.sqrt(area) // 15, 8), 2)
        target = target - rng.integers(-noise_strength,
                                       noise_strength+1,
                                       size=target.shape)
        segmentation = segmentation.reshape(-1, len(segmentation), 2)
        target = target.reshape(-1, len(target), 2)
        matches=list()

        for i in range(0,len(segmentation[0])):
            matches.append(cv2.DMatch(i,i,0))

        self.tps.estimateTransformation(target, segmentation, matches)

        distorted_image = self.tps.warpImage(np.array(image))
        return Image.fromarray(distorted_image)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target_ann = self._load_target(id)
        segmentation = np.array(target_ann['segmentation'][0]).reshape(-1, 2)
        area = target_ann['area']
        image_distorted = self._warp_image(image, segmentation, area)
        target = np.zeros(self.num_categories)
        idx = self.categories[target_ann['category_id']]
        target[idx] = 1

        if self.transform is not None:
            image = self.transform(image)
            image_distorted = self.transform(image_distorted)

        return image, image_distorted, target

    def __len__(self) -> int:
        return len(self.ids)
