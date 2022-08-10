methods="gradcam layercam"
models="resnet50 vit_b_32 vgg16_bn swin_t"
files="000000008532.mp4 000000009448.mp4 000000024919.mp4 000000148719.mp4"

for model in $models; do
    for method in $methods; do
        for file in $files; do
            python process_video_cam.py \
                --method ${method} \
                --model_name ${model} \
                --in_path ${file}
        done
    done
done