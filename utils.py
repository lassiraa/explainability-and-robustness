from math import dist
from typing import Any, Callable, Optional, Tuple, List
import os
import json

import numpy as np
import cv2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from PIL import Image


rng = np.random.default_rng(51)


def calculate_perpendicular_translation(
    idx: int,
    segmentation: np.ndarray,
    magnitude: float
    ) -> np.ndarray:
    point = segmentation[idx,:]
    prev = segmentation[idx-1,:]
    next_idx = idx + 1 \
        if idx < segmentation.shape[0] - 1 \
        else 0
    next = segmentation[next_idx,:]
    v = (point - prev) + (point - next)
    norm = np.sqrt(np.sum(v**2))
    if norm == 0:
        return point
    return point + np.rint((v / norm) * magnitude)


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
        transforms: Optional[Callable] = None,
        distort_background: Optional[str] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.distort_background = distort_background
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
        image: np.ndarray,
        target_ann: dict,
        skip_every: int
    ) -> np.ndarray:
        
        segmentation = np.array(target_ann['segmentation'][0]).reshape(-1, 2)
        #  Following is for test purposes in example visualization
        # print(self.coco.loadCats(target_ann['category_id']))
        # for i in range(segmentation.shape[0]):
        #     x, y = list(segmentation[i,:])
        #     cv2.circle(image, (int(x), int(y)), 3, [255, 0, 0])
        area = target_ann['area']
        target = segmentation.copy()
        noise_strength = max(min(np.sqrt(area) // 20, 6), 1)
        len_targets = target.shape[0]
        for idx in range(len_targets):
            if idx % skip_every != 0:
                continue
            target[idx,:] = calculate_perpendicular_translation(
                idx,
                target,
                noise_strength
            )
        # target = target - rng.integers(-noise_strength,
        #                                noise_strength+1,
        #                                size=target.shape)
        segmentation = segmentation.reshape(-1, len(segmentation), 2)
        target = target.reshape(-1, len(target), 2)
        
        matches=[]

        for i in range(0,len(segmentation[0])):
            matches.append(cv2.DMatch(i,i,0))

        self.tps.estimateTransformation(target, segmentation, matches)

        distorted_image = self.tps.warpImage(image)
        return distorted_image

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        image = np.array(image)
        target_ann = self._load_target(id)

        if self.distort_background is not None:
            mask = self.coco.annToMask(target_ann)
            if self.distort_background == 'blur':
                image_blurred = cv2.GaussianBlur(image, (21, 21), 0.5*((21-1)*0.5 - 1) + 0.8)
                image = np.where(mask[..., None] == 1, image, image_blurred)
            if self.distort_background == 'remove':
                image = image * mask[..., None]

        image_distorted = self._warp_image(image, target_ann, 4)
        target = np.zeros(self.num_categories)
        idx = self.categories[target_ann['category_id']]
        target[idx] = 1

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))
            image_distorted = self.transform(Image.fromarray(image_distorted))

        return image, image_distorted, target

    def __len__(self) -> int:
        return len(self.ids)
