import numpy as np
import matplotlib.pyplot as plt
import requests
from pycocotools.coco import COCO
import cv2
from transformers import AutoFeatureExtractor, ViTForImageClassification

from TPSwarping import WarpImage_TPS


rng = np.random.default_rng(51)


def load_and_tps_warp(img_id, coco_annotation):
    img_info = coco_annotation.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]
    img_url = img_info["coco_url"]

    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)
    mask = coco_annotation.annToMask(anns[0])

    # Use URL to load image.
    resp = requests.get(img_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Blur background
    blurred_im = cv2.GaussianBlur(im, (21, 21), 0)
    #im = im * mask[..., None]
    im = np.where(mask[..., None] == 1, im, blurred_im)

    # Get segmentation points (x,y) and draw them to the image.
    segmentation = np.array(anns[0]['segmentation'][0]).reshape(-1, 2)
    # for i in range(segmentation.shape[0]):
    #     x, y = list(segmentation[i,:])
    #     cv2.circle(im, (int(x), int(y)), 3, [255, 0, 0])
    cv2.imshow('Original image', im)
    cv2.waitKey(0)

    # Make target for TPS to be randomly added noise around segmentation points.
    target = segmentation.copy()
    noise_strength = min(np.sqrt(anns[0]['area']) // 15, 10)
    target = target - rng.integers(-noise_strength,
                                   noise_strength+1,
                                   size=target.shape)

    # Warp image using source and target points utilizing thin plate spline.
    new_im, new_pts1, new_pts2 = WarpImage_TPS(segmentation, target, im)

    # Show resulting distorted image.
    cv2.imshow('Image with distorted shape', new_im)
    cv2.waitKey(0)
    return im, new_im


def classify_image(feature_extractor, model, image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

if __name__ == '__main__':
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-patch16-224')
    model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224')

    coco_annotation_file_path = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]

    # Iterate through categories
    for i, query_name in enumerate(cat_names):
        # Get category ID
        query_id = coco_annotation.getCatIds(catNms=[query_name])[0]

        # Get the ID of all the images containing the object of the category.
        img_ids = coco_annotation.getImgIds(catIds=[query_id])

        # Iterate through images within category
        for img_i, img_id in enumerate(img_ids):
            # Load and show image and the resulting warped image
            im, dist_im = load_and_tps_warp(img_id, coco_annotation)
            classify_image(feature_extractor, model, im)
            classify_image(feature_extractor, model, dist_im)
