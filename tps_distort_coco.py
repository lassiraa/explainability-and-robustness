import numpy as np
import matplotlib.pyplot as plt
import requests
from pycocotools.coco import COCO
import cv2

from TPSwarping import WarpImage_TPS

if __name__ == '__main__':
    coco_annotation_file_path = "/media/lassi/Data/datasets/coco/annotations/instances_val2017.json"

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]

    # Category ID -> Category Name.
    query_id = cat_ids[0]
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]

    # Category Name -> Category ID.
    query_name = cat_names[2]
    query_id = coco_annotation.getCatIds(catNms=[query_name])[0]

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds(catIds=[query_id])

    # Pick one image.
    img_id = img_ids[2]
    img_info = coco_annotation.loadImgs([img_id])[0]
    img_file_name = img_info["file_name"]
    img_url = img_info["coco_url"]

    # Get all the annotations for the specified image.
    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)

    # Use URL to load image.
    resp = requests.get(img_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)

    segmentation = np.array(anns[0]['segmentation'][0]).reshape(-1, 2)
    for i in range(segmentation.shape[0]):
        x, y = list(segmentation[i,:])
        cv2.circle(im, (int(x), int(y)), 3, [255, 0, 0])
    cv2.imshow('Original image', im)
    cv2.waitKey(0)

    source = segmentation.copy()
    target = segmentation.copy()
    target = target - np.random.randint(-5, 6, size=target.shape)

    new_im, new_pts1, new_pts2 = WarpImage_TPS(source, target, im)
    cv2.imshow('Image with distorted shape', new_im)
    cv2.waitKey(0)