from math import dist
from typing import Any, Callable, Optional, Tuple, List
import os
import json

import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from PIL import Image
from skimage.segmentation import find_boundaries


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


def calculate_mass_within(
    saliency_map: torch.tensor,
    class_mask: torch.tensor
) -> float:
    class_mask = class_mask
    mass = saliency_map.sum()
    mass_within = (saliency_map * class_mask).sum()
    return (mass_within / mass).item()


def reshape_transform_vit(tensor):
    dim = int(np.sqrt(tensor[:, 1:, :].shape[1]))
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      dim, dim, tensor.size(2))

    #  Bring the channels to the first dimension,
    #  like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


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
        distortion_method: str = 'perpendicular',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        distort_background: Optional[str] = None,
        debug_mode: bool = False
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.distort_background = distort_background
        self.distortion_method = distortion_method
        self.debug_mode = debug_mode
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
    
    def _smooth_transition(
        self,
        image: np.ndarray,
        image_distorted: np.ndarray,
        mask: np.ndarray,
        steps: int,
        kernel_size: int
    ) -> np.ndarray:
        step_multiplier = 1 / (steps + 1)
        kernel = np.ones((kernel_size+2, kernel_size+2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        result = image_distorted * mask[..., None]
        
        #  Dilate the mask every step
        for i in range(steps):
            multiplier = step_multiplier * (i + 1)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            mask_diff = mask_dilated - mask
            #  Make linear combination of distorted and not distorted image
            #  in the zone of the dilated part of image
            add = mask_diff[..., None] * (image_distorted * (1 - multiplier) + multiplier * image)
            result += add.astype(np.uint8)
            mask = mask_dilated
        
        #  Fill rest of image with original image
        result = np.where(mask[..., None] == 1, result, image)
        return result
    
    def _warp_image(
        self,
        image: np.ndarray,
        target_ann: dict,
        skip_every: int
    ) -> np.ndarray:
        
        segmentation = np.array(target_ann['segmentation'][0]).reshape(-1, 2)
        #  Following is for test purposes in example visualization
        if self.debug_mode:
            print(self.coco.loadCats(target_ann['category_id']))
            # for i in range(segmentation.shape[0]):
            #     x, y = list(segmentation[i,:])
            #     cv2.circle(image, (int(x), int(y)), 3, [255, 0, 0])
        area = target_ann['area']
        target = segmentation.copy()
        noise_strength = max(min(np.sqrt(area) // 15, 8), 2)
        len_targets = target.shape[0]

        #  Add noise to target by chosen method

        if self.distortion_method == 'perpendicular':
            for idx in range(len_targets):
                if idx % skip_every != 0:
                    continue
                target[idx,:] = calculate_perpendicular_translation(
                    idx, target, noise_strength
                )

        if self.distortion_method == 'random_noise':

            for idx in range(len_targets):
                if idx % skip_every != 0:
                    continue
                target[idx,:] += rng.integers(
                    -noise_strength,
                    noise_strength+1,
                    size=2)

        if self.distortion_method == 'random_walk':
            prev_target = None
            for idx in range(len_targets):
                if prev_target is None:
                    target[idx,:] = calculate_perpendicular_translation(
                        idx, target, noise_strength+2
                    )
                    prev_target = segmentation[idx,:]
                    continue
                direction = segmentation[idx,:] - prev_target
                noise = rng.integers(-1, 2, size=2)
                target[idx,:] = target[idx-1,:] + direction + noise
                prev_target = segmentation[idx,:]
        
        if self.distortion_method == 'singular_spike':
            random_idx = rng.integers(0, target.shape[0])
            target[random_idx,:] = calculate_perpendicular_translation(
                random_idx, target, noise_strength*2
            )
        
    
        segmentation = segmentation.reshape(-1, len(segmentation), 2)
        target = target.reshape(-1, len(target), 2)
        
        matches=[]

        for i in range(0,len(segmentation[0])):
            matches.append(cv2.DMatch(i,i,0))

        self.tps.estimateTransformation(target, segmentation, matches)

        image_distorted = self.tps.warpImage(image)
        return image_distorted

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

        if self.distort_background == 'smooth_transition':
            image_distorted = self._smooth_transition(
                image=image,
                image_distorted=image_distorted,
                mask=mask,
                steps=6,
                kernel_size=5
            )
        
        target = np.zeros(self.num_categories)
        idx = self.categories[target_ann['category_id']]
        target[idx] = 1

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))
            image_distorted = self.transform(Image.fromarray(image_distorted))

        return image, image_distorted, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoExplainabilityMeasurement(VisionDataset):
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
        class_to_targets = dict()

        for ann in anns:
            if 'category_id' not in ann:
                continue
            
            mask = self.coco.annToMask(ann).astype(bool)
            idx = self.categories[ann['category_id']]
            
            if idx in class_to_targets:
                class_to_targets[idx]['mask'] += mask
                continue
            
            labels = np.zeros(self.num_categories, dtype=np.float32)
            labels[idx] = 1
            class_to_targets[idx] = dict(
                mask=mask,
                labels=labels
            )
        
        return class_to_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        class_to_targets = self._load_target(id)

        #  Transform both the masks and the images.
        #  If one is transformed, the other one needs to be as well.
        if self.transform is not None and self.target_transform is not None:
            image = self.transform(image)
            class_to_targets = {
                idx: {
                    'mask': self.target_transform(target['mask'].astype('float')),
                    'labels': target['labels']
                }
                for idx, target in class_to_targets.items()
            }

        return image, class_to_targets

    def __len__(self) -> int:
        return len(self.ids)
