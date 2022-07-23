from typing import Any, Callable, Optional, Tuple, List
import os
import json

import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from PIL import Image
from skimage.segmentation import find_boundaries


rng = np.random.default_rng(51)


def load_model_with_target_layers(
    model_name: str,
    device: torch.device
) -> nn.Module:
    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, 80)
        target_layers = [model.features[-1]]

    if model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
        in_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(in_features, 80)
        target_layers = [model.encoder.layers[-1].ln_1]

    if model_name == 'swin_t':
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 80)
        target_layers = [model.features[-1][-1].norm1]

    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 80)
        target_layers = [model.layer4[-1]]
    
    model.load_state_dict(torch.load(f'{model_name}_coco.pt'))
    model.to(device)
    return model, target_layers


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


def reshape_transform_vit(tensor, dim=7):
    #  Needed for ViT but not for Swin
    if tensor.shape[1] == (dim * dim + 1):
        tensor = tensor[:, 1:, :]
    
    result = tensor.reshape(tensor.size(0),
                            dim, dim, -1)

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
 
    def _load_targets(self, id: int) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        labels = np.zeros(self.num_categories, dtype=np.float32)

        for ann in anns:
            assert('category_id' in ann)
            idx = self.categories[ann['category_id']]
            labels[idx] = 1
        
        return labels
    
    def _smooth_transition(
        self,
        vector_field: np.ndarray,
        mask: np.ndarray,
        steps: int,
        kernel_size: int
    ) -> np.ndarray:
        step_multiplier = 1 / (steps + 1)
        kernel = np.ones((kernel_size+2, kernel_size+2), np.float32)
        mask = cv2.dilate(mask, kernel, iterations=1)
        result = vector_field * mask
        
        #  Dilate the mask every step
        for i in range(steps):
            multiplier = step_multiplier * (i + 1)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            mask_diff = mask_dilated - mask
            #  Make linear combination of distorted and not distorted vector fields
            add = mask_diff * vector_field * (1 - multiplier)
            result += add.astype(np.float32)
            mask = mask_dilated
    
        #  Fill rest of image with original image
        result = np.where(mask == 1, result, 0)
        return result
    
    def _warp_image(
        self,
        image: np.ndarray,
        target_ann: dict,
        mask: np.ndarray,
        skip_every: int
    ) -> np.ndarray:
        segmentation = np.array(target_ann['segmentation'][0]).reshape(-1, 2)

        #  Following is for test purposes in example visualization
        if self.debug_mode:
            print(self.coco.loadCats(target_ann['category_id']))
            for i in range(segmentation.shape[0]):
                x, y = list(segmentation[i,:])
                cv2.circle(image, (int(x), int(y)), 3, [255, 0, 0])
        
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

        indices = np.float32(
            np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        ).T.reshape(1,-1,2)

        #  Apply transformation to all (col, row) indices
        _, new_indices = self.tps.applyTransformation(np.float32(indices))

        #  Create vector field of the point transitions
        map_x = new_indices[0,:,0].reshape(image.shape[0], image.shape[1], order='F')
        ind_x = indices[0,:,0].reshape(image.shape[0], image.shape[1], order='F')
        vector_field_x = map_x - ind_x
        #  Smooth transition of vector field around the object
        vector_field_x_smoothed = self._smooth_transition(
            vector_field=vector_field_x,
            mask=mask,
            steps=5,
            kernel_size=9
        )
        map_x = vector_field_x_smoothed + ind_x
        
        #  Repeat smoothing process for y-coordinate
        map_y = new_indices[0,:,1].reshape(image.shape[0], image.shape[1], order='F')
        ind_y = indices[0,:,1].reshape(image.shape[0], image.shape[1], order='F')
        vector_field_y = map_y - ind_y
        vector_field_y_smoothed = self._smooth_transition(
            vector_field=vector_field_y,
            mask=mask,
            steps=5,
            kernel_size=9
        )
        map_y = vector_field_y_smoothed + ind_y

        #  Apply remap with x and y coordinate mappings to obtain thinplate spline result
        image_distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return image_distorted

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        image = np.array(image)
        target_ann = self._load_target(id)
        all_labels = self._load_targets(id)
        mask = self.coco.annToMask(target_ann)

        if self.distort_background == 'blur':
            image_blurred = cv2.GaussianBlur(image, (21, 21), 0.5*((21-1)*0.5 - 1) + 0.8)
            image = np.where(mask[..., None] == 1, image, image_blurred)
        if self.distort_background == 'remove':
            image = image * mask[..., None]

        image_distorted = self._warp_image(image, target_ann, mask, 4)
        
        distorted_label = np.zeros(self.num_categories)
        idx = self.categories[target_ann['category_id']]
        distorted_label[idx] = 1

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))
            image_distorted = self.transform(Image.fromarray(image_distorted))

        return image, image_distorted, distorted_label, all_labels

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
