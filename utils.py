from typing import Any, Callable, Optional, Tuple, List
import os

import numpy as np
import cv2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from PIL import Image


rng = np.random.default_rng(51)


class CocoToKHot(object):
    def __init__(self, path2json):
        categories = COCO(path2json).getCatIds()
        self.num_categories = len(categories)
        #  Turn categories list into category_id: index dictionary
        self.categories = {cat: idx for idx, cat in enumerate(categories)}
    
    def __call__(self, anns):
        labels = np.zeros(self.num_categories, dtype=np.float32)

        for ann in anns:
            assert('category_id' in ann)
            idx = self.categories[ann['category_id']]
            labels[idx] = 1
        
        return labels


class CocoDistortion(VisionDataset):
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
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def _warp_image(self, image: Image.Image, segmentation: np.ndarray, area: float):
        target = segmentation.copy()
        noise_strength = min(np.sqrt(area) // 15, 10)
        target = target - rng.integers(-noise_strength,
                                       noise_strength+1,
                                       size=target.shape)
        segmentation = segmentation.reshape(-1, len(segmentation), 2)
        target = target.reshape(-1, len(target), 2)
        matches=list()

        for i in range(0,len(segmentation[0])):
            matches.append(cv2.DMatch(i,i,0))

        self.tps.estimateTransformation(target, segmentation, matches)

        distorted_image = self.tps.warpImage(image)
        return distorted_image


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        anns = self._load_target(id)
        # TODO: Add selection criteria for chosen segmentation
        target_ann = anns[0]
        segmentation = np.array(target_ann['segmentation'][0]).reshape(-1, 2)
        area = target_ann['area']
        image_distorted = self._warp_image(image, segmentation, area)

        if self.transform is not None:
            image = self.transform(image)
            image_distorted = self.transform(image_distorted)

        if self.target_transform is not None:
            target = self.target_transform(target_ann)

        return image, image_distorted, target


    def __len__(self) -> int:
        return len(self.ids)
