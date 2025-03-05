import albumentations as A
import cv2
from pycocotools.coco import COCO
import os
import json
import random
from tqdm import tqdm


"""
This will carry on data augmentation on a folder of images.
FIXME no annotations can be augmented yet using the same image transformations. This part is 
commented out.
"""


def correct_bbox(bbox, image_width, image_height):
    """
    Corrects the bbox to ensure it has positive dimensions and is within image bounds.
    Args:
        bbox (list): Bounding box with format [x, y, width, height].
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    Returns:
        list: Corrected bounding box.
    """
    x, y, width, height = bbox
    width = abs(width)
    height = abs(height)

    # Adjust to ensure bbox is within image bounds
    x = max(0, min(x, image_width - width))
    y = max(0, min(y, image_height - height))

    return [x, y, width, height]

# Paths
image_dir = '/home/mrt/dev/detr/data/ms/person+car/all_images'
annotation_file = '/home/mrt/dev/detr/data/ms/person+car/annotations/cleaned_person_car_coco.json'
output_dir = '/home/mrt/dev/detr/data/ms/person+car/augmented/images'
output_annotation_file = '/home/mrt/dev/detr/data/ms/person+car/annotations/augmented_instances.json'

# Ensure the output directories exist
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(annotation_file)

# Define transformations
base_transform = A.Compose([
    A.SmallestMaxSize(max_size=800),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

additional_transforms = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.RandomRotate90(p=1),
    # Add more transformations if needed
]

# Initialize new annotations dict
new_annotations = {
    "images": [],
    "annotations": [],
    "categories": coco.dataset['categories']  # Copy categories from original annotations
}

image_id = 1  # Start with a new image id
annotation_id = 1  # Start with a new annotation id

for img_id in tqdm(list(coco.imgs.keys())):
    filename = coco.imgs[img_id]['file_name']
    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #ann_ids = coco.getAnnIds(imgIds=img_id)
    #annotations = coco.loadAnns(ann_ids)
    #bboxes = [ann['bbox'] for ann in annotations]
    #category_ids = [ann['category_id'] for ann in annotations]

    for transform in [base_transform] + additional_transforms:
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        transformed_image = transformed['image']
        #transformed_bboxes = transformed['bboxes']

        #transformed_bboxes = []
        #for bbox in transformed['bboxes']:
        #    corrected_bbox = correct_bbox(bbox, transformed['image'].shape[1], transformed['image'].shape[0])
        #    transformed_bboxes.append(corrected_bbox)

        #transformed_category_ids = transformed['category_ids']

        # Save augmented image
        aug_image_path = os.path.join(output_dir, f"{image_id}.jpg")
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_image_path, transformed_image)

        # Update new annotations
        #new_annotations['images'].append({
        #    "id": image_id,
        #    "file_name": f"{image_id}.jpg",
        #    "width": transformed_image.shape[1],
        #    "height": transformed_image.shape[0]
        #})

        #for bbox, category_id in zip(transformed_bboxes, transformed_category_ids):
        #    new_annotations['annotations'].append({
        #        "id": annotation_id,
        #        "image_id": image_id,
        #        "bbox": bbox,
        #        "category_id": category_id,
        #        "area": bbox[2] * bbox[3],
        #        "iscrowd": 0
        #    })
        #    annotation_id += 1

        image_id += 1

# Save new annotations to file
with open(output_annotation_file, 'w') as f:
    json.dump(new_annotations, f)

print("Augmentation completed.")
