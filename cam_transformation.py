from typing import Any

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import resized_crop
import numpy as np
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
from scipy.stats import spearmanr
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils import reshape_transform_vit, load_model_with_target_layers, scale_image


def process_saliency(
    input: torch.tensor,
    saliency_method: Any,
    class_idx: Any,
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create CAM visualization of video for highest prob. class')
    parser.add_argument('--in_path', type=str, required=True,
                        help='path to image')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for cam methods')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
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
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, target_layers = load_model_with_target_layers(args.model_name, device)

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
    rgb_img = cv2.imread(f'{args.in_path}', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input = image_transform(rgb_img)
    rgb_img = np.moveaxis(input.numpy(), 0, -1)
    i, j, h, w = transforms.RandomResizedCrop.get_params(input, scale=(0.75, 0.75), ratio=(1,1))
    print(i, j, h, w)
    cropped_input = resized_crop(input, i, j, h, w, size=(224,224))
    rgb_img_cropped = np.moveaxis(cropped_input.numpy(), 0, -1)
    input = image_normalize(input)
    input = input.to(device=device, dtype=torch.float32).unsqueeze(0)
    cropped_input = image_normalize(cropped_input)
    cropped_input = cropped_input.to(device=device, dtype=torch.float32).unsqueeze(0)
    class_idx = model(input).argmax().item()

    #  Process saliency map
    # start = time.time()
    # for _ in range(10):
    #     saliency_map = process_saliency(input, saliency_method, args, is_backprop)
    # time_taken = time.time() - start
    # print(f'Time for 10 iterations using {args.method}/{args.model_name}: {time_taken:.3f}s')
    original_saliency = process_saliency(input, saliency_method, class_idx, is_backprop)
    saliency_map = resized_crop(torch.from_numpy(original_saliency[None,:,:]), i, j, h, w, size=(224,224)).numpy().reshape(224, 224)
    cropped_saliency_map = process_saliency(cropped_input, saliency_method, class_idx, is_backprop)
    correlation = spearmanr(saliency_map, cropped_saliency_map, axis=None)
    print(correlation)


    cam_image_transformed_image = show_cam_on_image(rgb_img_cropped, cropped_saliency_map, use_rgb=True)
    cam_image_transformed_image = cv2.cvtColor(cam_image_transformed_image, cv2.COLOR_RGB2BGR)

    cam_image_transformed_saliency = show_cam_on_image(rgb_img_cropped, saliency_map, use_rgb=True)
    cam_image_transformed_saliency = cv2.cvtColor(cam_image_transformed_saliency, cv2.COLOR_RGB2BGR)
    rgb_img = cv2.cvtColor(rgb_img*255, cv2.COLOR_RGB2BGR)
    cv2.imwrite('original_image.jpg', rgb_img)
    rgb_img = cv2.rectangle(rgb_img, (j,i), (j+w,i+h), thickness=2, color=(0,0,255))
    original_saliency = cv2.rectangle(original_saliency, (j,i), (j+w,i+h), thickness=2, color=(0,0,255))
    rgb_img_cropped = cv2.cvtColor(rgb_img_cropped*255, cv2.COLOR_RGB2BGR)
    cv2.imwrite('image_with_crop_rectangle.jpg', rgb_img)
    cv2.imwrite('cropped_image.jpg', rgb_img_cropped)
    cv2.imwrite('original_saliency.jpg', original_saliency*255)
    cv2.imwrite('cropped_image_saliency.jpg', cropped_saliency_map*255)
    cv2.imwrite('cropped_saliency_saliency.jpg', saliency_map*255)
    cv2.imwrite(f'transformed_image.jpg', cam_image_transformed_image)
    cv2.imwrite(f'transformed_saliency.jpg', cam_image_transformed_saliency)