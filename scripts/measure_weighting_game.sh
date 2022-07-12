# Directory of validation images
images_dir="/media/lassi/Data/datasets/coco/images/val2017/"
ann_path="/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"

# Combination of methods and models to try
methods="gradcam gradcam++ xgradcam layercam guidedbackprop ablationcam"
models="resnet50 vit_b_32 vgg16"


for model in $models; do
    for method in $methods; do
        python weighting_game.py \
            --method ${method} \
            --model ${model} \
            --images_dir ${images_dir} \
            --ann_path ${ann_path}
    done
done