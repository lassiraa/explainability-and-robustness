from typing import Callable

import moviepy.editor as mpy 
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy import stats
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

from utils import reshape_transform_vit, load_model_with_target_layers


def process_video(
    video: mpy.VideoFileClip,
    device: torch.device,
    saliency_method: Callable,
    class_idx: int,
    image_transform: transforms.Compose,
    image_normalize: transforms.Normalize
) -> tuple[np.ndarray]:
    frames = []
    corrs = []
    prev_saliency = None

    for frame in video.iter_frames():
        #  Preprocess frame for network
        frame = image_transform(frame / 255)
        input = image_normalize(frame).to(device=device, dtype=torch.float32).unsqueeze(0)
        #  Keep numpy version of frame for cam visualization
        frame_npy = np.float32(frame.numpy())
        frame_npy = np.moveaxis(frame_npy, 0, -1)

        #  Process saliency map
        saliency_map = saliency_method(input, [ClassifierOutputTarget(class_idx)])[0, :]
        cam_frame = show_cam_on_image(
            img=frame_npy,
            mask=saliency_map,
            use_rgb=True
        )
        if prev_saliency is not None:
            corr, _ = stats.spearmanr(prev_saliency, saliency_map, axis=None)
            corrs.append(corr)
        else:
            corrs.append(0)
        
        prev_saliency = np.copy(saliency_map)
        #  Add frame to cam visualization video
        frames.append(cam_frame)
    
    return frames, corrs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create CAM visualization of video for highest prob. class')
    parser.add_argument('--in_path', type=str, required=True,
                        help='path to input video')
    parser.add_argument('--class_idx', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for cam methods')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='workers for dataloader')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='name of model used for inference',
                        choices=[
                            'vit_b_32', 'vit_b_16', 'vit_l_32', 'vit_l_16',
                            'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn',
                            'resnet50', 'resnet101', 'resnet152', 'swin_t'
                        ])
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
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
    video = mpy.VideoFileClip(f'./stability_videos/{args.in_path}')
    fname = args.in_path.split('.')[0]
    start_frame = image_transform(video.get_frame(0)).to(device).unsqueeze(0)
    start_output = model(start_frame)
    #class_idx = start_output.argmax().item()
    class_idx = args.class_idx

    cam_frames, corrs = process_video(
        video=video,
        device=device,
        saliency_method=saliency_method,
        class_idx=class_idx,
        image_transform=image_transform,
        image_normalize=image_normalize
    )
    
    mean_corr = np.nanmean(corrs)
    corr_str = str(round(mean_corr*1000))

    out_video = mpy.ImageSequenceClip(cam_frames, fps=25)
    out_video.write_videofile(f'./stability_videos/{fname}_{args.method}_{args.model_name}_{class_idx}_{corr_str}.mp4')