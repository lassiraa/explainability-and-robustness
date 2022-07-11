from typing import Callable
import os
import json

import moviepy.editor as mpy 
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy import stats
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GuidedBackpropReLUModel
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import reshape_transform_vit, load_model_with_target_layers


def calculate_mean_correlation(
    video: mpy.VideoFileClip,
    device: torch.device,
    saliency_method: Callable,
    class_idx: int,
    image_transform: transforms.Compose,
    image_normalize: transforms.Normalize,
    is_backprop: bool
) -> tuple[np.ndarray]:
    correlations = []
    prev_saliency = None

    for i, frame in enumerate(video.iter_frames()):
        
        #  Only process first 10 frames and last 10 frames
        if i > 10 and i < 139:
            prev_saliency = None
            continue

        #  Preprocess frame for model
        frame = image_transform(frame / 255)
        input = image_normalize(frame).to(device=device, dtype=torch.float32).unsqueeze(0)

        #  Process saliency map
        if is_backprop:
            saliency_map = saliency_method(input, target_category=class_idx)
        else:
            saliency_map = saliency_method(input, [ClassifierOutputTarget(class_idx)])[0, :]

        #  Calculate Spearman correlation of saliency map from previous frame to current frame
        if prev_saliency is not None:
            corr, _ = stats.spearmanr(prev_saliency, saliency_map, axis=None)
            correlations.append(corr)

        prev_saliency = np.copy(saliency_map)
    
    return np.mean(correlations)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Measure stability of explanation method')
    parser.add_argument('--in_path', type=str, required=True,
                        help='path folder containing videos')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for cam methods')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='name of model used for inference',
                        choices=[
                            'vit_b_32', 'vit_b_16', 'vit_l_32', 'vit_l_16',
                            'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn',
                            'resnet50', 'resnet101', 'resnet152'
                        ])
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
    is_vit = 'vit_' == args.model_name[:4]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_vit:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        reshape_transform = reshape_transform_vit

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
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
    
    corrs = []
    for fname in tqdm(os.listdir(args.in_path)):
        video = mpy.VideoFileClip(args.in_path+fname)
        #  Use highest probability class from first frame for CAM
        start_frame = image_transform(video.get_frame(0)).to(device).unsqueeze(0)
        start_output = model(start_frame)
        class_idx = start_output.argmax().item()

        corr = calculate_mean_correlation(
            video=video,
            device=device,
            saliency_method=saliency_method,
            class_idx=class_idx,
            image_transform=image_transform,
            image_normalize=image_normalize,
            is_backprop=is_backprop
        )
    
        corrs.append(corr)
    
    #  Save results to json
    with open(f'data/{args.model_name}_{args.method}_stability.json', 'w') as fp:
        json.dump(corrs, fp)