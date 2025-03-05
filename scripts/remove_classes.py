'''
To filter the COCO validation dataset to keep only the images and annotations for the classes you've specified (person, bicycle, car, etc.), you'll need to write a script that processes the COCO annotations JSON file. The script will:

    1. Load the original COCO annotations.
    2. Filter out annotations that do not belong to the specified classes.
    3. Save a new JSON file with the filtered annotations.
    4. Optionally, move or copy the images corresponding to the remaining annotations to a new directory.
'''
from pycocotools.coco import COCO
import json
import shutil
import os

# Path to the original COCO annotations and images
annotations_path = '/home/mrt/dev/detr/data/tmp/merged_annotations.json'
images_path = '/home/mrt/dev/detr/data/tmp/images'

# Path to save the filtered annotations and images
filtered_annotations_path = '/home/mrt/dev/detr/data/tmp/filtered_annotations.json'
filtered_images_path = '/home/mrt/dev/detr/data/tmp/filtered_images'

# Initialize COCO API
coco = COCO(annotations_path)

# IDs of categories to keep
categories_to_keep = [1, 3, 10]

# Get ID of images that have the categories we're interested in
img_ids = []
for cat_id in categories_to_keep:
    img_ids.extend(coco.getImgIds(catIds=[cat_id]))
# Remove duplicate image IDs
img_ids = list(set(img_ids))

# Load images and annotations for the selected image IDs
images = coco.loadImgs(img_ids)
annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_ids, catIds=categories_to_keep))

# Filtered dataset to save
filtered_dataset = {
    "images": images,
    "annotations": annotations,
    "categories": coco.loadCats(categories_to_keep)
}

# Save the filtered annotations to a new JSON file
with open(filtered_annotations_path, 'w') as f:
    json.dump(filtered_dataset, f)

# Optionally, copy the images to a new directory
if not os.path.exists(filtered_images_path):
    os.makedirs(filtered_images_path)

for img in images:
    # Construct image file path
    filename = img['file_name']
    original_img_path = os.path.join(images_path, filename)
    new_img_path = os.path.join(filtered_images_path, filename)
    
    # Copy image
    shutil.copy(original_img_path, new_img_path)

print(f"Filtered annotations saved to {filtered_annotations_path}")
print(f"Filtered images copied to {filtered_images_path}")
