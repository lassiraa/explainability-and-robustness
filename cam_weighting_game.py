from typing import Any
import time

import torch
import torchvision.transforms as transforms
import numpy as np
from pycocotools.coco import COCO
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    LayerCAM, \
    FullGrad, \
    GuidedBackpropReLUModel
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from utils import reshape_transform_vit, load_model_with_target_layers, scale_image


def process_saliency(
    input: torch.tensor,
    saliency_method: Any,
    class_idx,
    is_backprop: bool,
) -> np.ndarray:
    if is_backprop:
        saliency_map = saliency_method(input, target_category=class_idx)
        saliency_map = saliency_map.sum(axis=2).reshape(224, 224)
        saliency_map = np.where(saliency_map > 0, saliency_map, 0)
        saliency_map = scale_image(saliency_map, 1)
    else:
        saliency_map = saliency_method(input, [ClassifierOutputTarget(class_idx)])[0, :]
    return saliency_map

def load_masks(coco, categories, id):
    anns = coco.loadAnns(coco.getAnnIds(id))
    class_to_targets = dict()

    for ann in anns:
        if 'category_id' not in ann:
            continue
        
        mask = coco.annToMask(ann).astype(bool)
        class_idx = categories[ann['category_id']]
        
        if class_idx in class_to_targets:
            class_to_targets[class_idx]['mask'] += mask
            continue
        
        class_to_targets[class_idx] = dict(
            mask=mask
        )
    
    return class_to_targets


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create CAM visualization of video for highest prob. class')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for cam methods')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='name of model used for inference',
                        choices=['vit_b_32', 'vgg16_bn', 'swin_t', 'resnet50'])
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad',
                                 'guidedbackprop'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--ann_path', type=str,
                        default='/media/lassi/Data/datasets/coco/annotations/instances_val2017.json',
                        help='path to root directory containing annotations')
    parser.add_argument('--images_dir', type=str,
                        default='/media/lassi/Data/datasets/coco/images/val2017/',
                        help='path to coco root directory containing image folders')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, target_layers = load_model_with_target_layers(args.model_name, device)
    coco = COCO(args.ann_path)
    categories = {cat: idx for idx, cat in enumerate(coco.getCatIds())}
    for img_id, img_info in coco.imgs.items():
        # if img_id != 511321:
        #     continue
        img_fname = img_info['file_name']
        img_path = args.images_dir + img_fname
        masks = load_masks(coco, categories, img_id)
        reshape_transform = None
        is_vit = args.model_name in ['vit_b_32', 'swin_t']

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if is_vit:
            reshape_transform = reshape_transform_vit

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])

        image_normalize = transforms.Normalize(mean=mean, std=std)

        methods = \
            {"gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "fullgrad": FullGrad,
            "layercam": LayerCAM,
            "guidedbackprop": GuidedBackpropReLUModel}

        method = methods[args.method]
        is_backprop = False
        if args.method == 'guidedbackprop':
            saliency_method = method(model=model,
                                    use_cuda=torch.cuda.is_available())
            is_backprop = True
        elif args.method == 'ablationcam' and is_vit:
            saliency_method = method(model=model,
                                    target_layers=target_layers,
                                    reshape_transform=reshape_transform,
                                    use_cuda=torch.cuda.is_available(),
                                    ablation_layer=AblationLayerVit())
            saliency_method.batch_size = args.batch_size
        else:
            saliency_method = method(model=model,
                                    target_layers=target_layers,
                                    reshape_transform=reshape_transform,
                                    use_cuda=torch.cuda.is_available())
            saliency_method.batch_size = args.batch_size
        
        #  Read video and find highest probability class from first frame.
        #  Class ID is used for CAM visualization
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input = image_transform(rgb_img)
        rgb_img = np.moveaxis(input.numpy(), 0, -1)
        input = image_normalize(input)
        input = input.to(device=device, dtype=torch.float32).unsqueeze(0)
        class_idx = model(input).argmax().item()

        #  Process saliency map
        # start = time.time()
        # for _ in range(10):
        #     saliency_map = process_saliency(input, saliency_method, args, is_backprop)
        # time_taken = time.time() - start
        # print(f'Time for 10 iterations using {args.method}/{args.model_name}: {time_taken:.3f}s')
        saliency_map = process_saliency(input, saliency_method, class_idx, is_backprop)
        if class_idx not in masks:
            continue

        mask = masks[class_idx]['mask']
        mask = image_transform(mask).numpy()
        mask = mask.astype(np.bool8)

        mask_dilated = cv2.dilate(mask.astype(np.float32), kernel=np.ones((9,9)), iterations=1)
        mask_dilated = mask_dilated.astype(np.bool8)
        
        mask_binary_dilated = mask_dilated.squeeze(0)
        mask_binary_dilated = mask_binary_dilated.astype(np.uint8)
        mask_binary_dilated = np.repeat(mask_binary_dilated[:, :, np.newaxis], 3, axis=2)
        mask_binary_dilated = cv2.threshold(mask_binary_dilated, 0.5, 255, cv2.THRESH_BINARY)[1]
        mask_binary_dilated = cv2.cvtColor(mask_binary_dilated, cv2.COLOR_BGR2GRAY)

        mask_binary = mask.squeeze(0)
        mask_binary = mask_binary.astype(np.uint8)
        mask_binary = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
        mask_binary = cv2.threshold(mask_binary, 0.5, 255, cv2.THRESH_BINARY)[1]
        mask_binary = cv2.cvtColor(mask_binary, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        saliency_weighting_game = (saliency_map * mask_dilated).squeeze(0)
        acc = (saliency_weighting_game.sum() / saliency_map.sum())
        if np.isnan(acc):
            continue
        acc_thousand = str(round(acc * 1000))
        hit = mask.flatten()[saliency_map.argmax()]
        idx = saliency_map.argmax()
        pointing_game_idx = (idx % saliency_map.shape[0], idx // saliency_map.shape[0])
        hit_str = 'hit' if hit else 'miss'

        cam_image = show_cam_on_image(rgb_img, saliency_map, use_rgb=True)
        cam_image_weighting_game = show_cam_on_image(rgb_img, saliency_weighting_game, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cam_image_weighting_game = cv2.cvtColor(cam_image_weighting_game, cv2.COLOR_RGB2BGR)
        rgb_img = cv2.cvtColor(rgb_img*255, cv2.COLOR_RGB2BGR)
        
        #if 0.5 < acc and hit and mask.sum()/(224*224) < 0.2:
        cv2.imwrite(f'./weighting_game_examples/{img_id}.jpg', rgb_img)

        cv2.drawContours(rgb_img, contours, -1, (0, 0, 255), 1)
        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_contours.jpg', rgb_img)

        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_mask.jpg', mask_binary)
        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_mask_dilated.jpg', mask_binary_dilated)

        rgb_img = cv2.circle(rgb_img, pointing_game_idx, 2, (0,255,0), thickness=2)
        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_pointing_game_{hit_str}.jpg', rgb_img)

        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_saliency.jpg', cam_image)
        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_weighting_game_{acc_thousand}.jpg', cam_image_weighting_game)
        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_saliency_mask.jpg', saliency_map*255)
        cv2.imwrite(f'./weighting_game_examples/{img_id}_{args.model_name}_weighting_game_mask_{acc_thousand}.jpg', saliency_weighting_game*255)
        
    