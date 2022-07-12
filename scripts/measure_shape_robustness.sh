# Directory of validation images
images_dir="/media/lassi/Data/datasets/coco/images/val2017/"
ann_path="/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"

# Models to try
models="resnet50 vit_b_32 vgg16"

# Other parameters
batch_size=100
num_workers=16


for model in $models; do
    python shape_robustness.py \
        --model ${model} \
        --images_dir ${images_dir} \
        --ann_path ${ann_path} \
        --batch_size ${batch_size} \
        --num_workers ${num_workers}
done