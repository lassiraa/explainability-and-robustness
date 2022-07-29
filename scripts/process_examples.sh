# Combination of methods and models to try
methods="gradcam gradcam++ xgradcam layercam guidedbackprop ablationcam"
models="vgg16_bn resnet50 vit_b_32 swin_t"


for model in $models; do
    for method in $methods; do
        python cam.py \
            --method ${method} \
            --model_name ${model} \
            --in_path examples/both.png \
            --class_idx 15

        python cam.py \
            --method ${method} \
            --model_name ${model} \
            --in_path examples/both.png \
            --class_idx 16

        python cam.py \
            --method ${method} \
            --model_name ${model} \
            --in_path examples/person_dog.png \
            --class_idx 0

        python cam.py \
            --method ${method} \
            --model_name ${model} \
            --in_path examples/person_dog.png \
            --class_idx 16

        python cam.py \
            --method ${method} \
            --model_name ${model} \
            --in_path examples/person_bike.jpg \
            --class_idx 1

        python cam.py \
            --method ${method} \
            --model_name ${model} \
            --in_path examples/person_bike.jpg \
            --class_idx 0
    done
done