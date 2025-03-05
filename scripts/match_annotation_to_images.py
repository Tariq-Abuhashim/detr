
import json
import os
from pathlib import Path

def match(coco_data, image_path):

    # Number of images
    num_images = len(coco_data['images'])
    print(f"Number of images: {num_images}")

    for image in coco_data['images']:
       file_str = image_path+'/'+image['file_name']
       file_pth = Path(file_str)
       if not file_pth.exists():
           print(file_str)
       

coco_path = '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/all_annotations.json'
image_path = '/media/mrt/Whale/data/detr/pcc5/person-car/all_images'
with open(coco_path, 'r') as f:
    coco_data = json.load(f)
match(coco_data, image_path)
