import numpy as np
from pycocotools.coco import COCO


class CocoToKHot(object):
    def __init__(self, path2json):
        categories = COCO(path2json).getCatIds()
        self.num_categories = len(categories)
        #  Turn categories list into category_id: k-hot index dictionary
        self.categories = {cat: idx for idx, cat in enumerate(categories)}
    
    def __call__(self, anns):
        labels = np.zeros(self.num_categories, dtype=np.float32)

        for ann in anns:
            assert('category_id' in ann)
            idx = self.categories[ann['category_id']]
            labels[idx] = 1
        
        return labels
