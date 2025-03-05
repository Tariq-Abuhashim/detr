import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO

# Paths to your data
data_dir = "man_ir/"
annotation_file = os.path.join(data_dir, 'cleaned_coco_annotations.json')
images_dir = os.path.join(data_dir, '')  # assuming images are in 'train' folder

# Initialize COCO api for instance annotations
coco = COCO(annotation_file)

# Load and display random samples
cat_ids = coco.getCatIds()  # get all categories
img_ids = coco.getImgIds()  # get all image IDs
print(img_ids)

for i in range(len(img_ids)):
    img_data = coco.loadImgs(img_ids[i])[0]
    print(img_data)
    image = Image.open(os.path.join(images_dir, img_data['file_name']))

    # Load and display instance annotations
    plt.imshow(image)
    ax = plt.gca()
    
    ann_ids = coco.getAnnIds(imgIds=img_data['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    for ann in anns:
        bbox = ann['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='r', linewidth=2)
        ax.add_patch(rect)
        cat_id = ann['category_id']
        cat_name = coco.loadCats(cat_id)[0]['name']
        plt.text(bbox[0], bbox[1] - 10, cat_name, color='r')

    plt.axis('off')
    plt.show()