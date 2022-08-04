# Directory of videos containing 3D effect (150 frames each)
video_dir="/media/lassi/Data/datasets/coco/3d-effect-videos/val2017/"
methods="gradcam gradcam++ xgradcam layercam guidedbackprop"
models="resnet50 vit_b_32 vgg16_bn swin_t"


for model in $models; do
    for method in $methods; do
        python explanation_stability_video.py \
            --method ${method} \
            --model ${model} \
            --in_path ${video_dir}
    done
done