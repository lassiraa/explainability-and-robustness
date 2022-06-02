import os
import json

from pycocotools.coco import COCO
from PIL import Image


def get_percent_within_center(
        center: dict,
        bb: dict
    ) -> float:
    #  Get bounding box of intersection
    x_left = max(center['x1'], bb['x1'])
    y_top = max(center['y1'], bb['y1'])
    x_right = min(center['x2'], bb['x2'])
    y_bottom = min(center['y2'], bb['y2'])

    #  If bounding box is out of frame
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    #  Calculate the percentage of the object's bounding box that is within center
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    object_area = (bb['x2'] - bb['x1']) * (bb['y2'] - bb['y1'])
    return intersection_area / object_area


if __name__ == '__main__':
    json_path = '/media/lassi/Data/datasets/coco/annotations/instances_val2017.json'
    img_folder = '/media/lassi/Data/datasets/coco/images/val2017/'
    coco = COCO(annotation_file=json_path)

    #  Dict for saving valid image ids and their chosen objects to distort
    img_to_ann = dict()
    
    for id in list(sorted(coco.imgs.keys())):
        anns = coco.loadAnns(coco.getAnnIds(id))
        
        #  Skip images with no objects
        if len(anns) == 0:
            continue

        #  Load image for it's dimensions
        fname = coco.loadImgs(id)[0]["file_name"]
        image = Image.open(os.path.join(img_folder, fname)).convert("RGB")
        #  Get width and height of image
        w, h = image.size
        #  Get center square bounding box
        bb_center_crop = dict()
        if w > h:
            margin = (w - h) // 2
            bb_center_crop['x1'] = margin
            bb_center_crop['x2'] = w - margin
            bb_center_crop['y1'] = 0
            bb_center_crop['y2'] = h
        else:
            margin = (h - w) // 2
            bb_center_crop['x1'] = 0
            bb_center_crop['x2'] = w
            bb_center_crop['y1'] = margin
            bb_center_crop['y2'] = h - margin
            
        #  Prefer largest objects
        for ann in sorted(anns, key=lambda x: x['area'], reverse=True):
            #  Check that object's bounding box is at least 75% in frame
            bb_object = dict()
            bb_object['x1'] = ann['bbox'][0]
            bb_object['x2'] = ann['bbox'][0] + ann['bbox'][2]
            bb_object['y1'] = ann['bbox'][1]
            bb_object['y2'] = ann['bbox'][1] + ann['bbox'][3]
            per = get_percent_within_center(bb_center_crop, bb_object)
            #  Also check proper formatted segmentation exists for annotation
            if per > 0.75 and len(ann['segmentation']) > 0 \
                    and isinstance(ann['segmentation'], list):
                img_to_ann[id] = ann['id']
    
    #  Save image to annotation dictionary as json
    with open('data/image_to_annotation.json', 'w') as fp:
        json.dump(img_to_ann, fp)
