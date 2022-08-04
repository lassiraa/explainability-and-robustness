# Directory of validation image
images_dir="/media/lassi/Data/datasets/coco/images/val2017/"
ann_path="/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"

methods="gradcam gradcam++ xgradcam layercam guidedbackprop"
models="resnet50 vit_b_32 vgg16_bn swin_t"


for model in $models; do
    for method in $methods; do
        python explanation_stability_transformation.py \
            --method ${method} \
            --model ${model} \
            --images_dir ${images_dir} \
            --ann_path ${ann_path}
    done
done