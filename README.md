# shape-robustness
Shape robustness exploration for ViT/CNN architectures

- fine_tune.py for fine tuning selected model to COCO dataset via multilabel classification. Currently supports vit_b_32 and vgg16 from torchvision pretrained models.
- shape_robustness.py contains script to evaluate shape robustness of fine tuned model.
- visualize_examples.py contains script to visualize random examples of images as how they would show in shape robustness evaluation.
- get_image_to_annotation.py has script that generates json in form {image_id: annotation id}, which contains all valid images and their chosen annotations. Selection criteria takes largest segmentation mask that is at least 75% within the center square of the image.
- utils.py contains Pytorch dataset implementations for training and shape robustness evaluation.