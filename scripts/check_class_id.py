from pycocotools.coco import COCO
import os

# Initialize COCO API for instance annotations (replace with the path to your annotations file)
annFile = '/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json'
coco = COCO(annFile)

# Get all categories and their IDs
cats = coco.loadCats(coco.getCatIds())

# Print category names and their corresponding IDs
print('COCO categories and IDs:')
for cat in cats:
    print(f'ID: {cat["id"]} Name: {cat["name"]}')
