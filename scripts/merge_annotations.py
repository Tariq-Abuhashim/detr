''' 
Integrating two annotation files and their corresponding image sets into a single dataset can be made more general to accommodate a variety of use cases. 

Note: in this code, dataset2 is added to dataset1, so its new category counting starts from last dataset1 category + 1

    1. Dynamically find the maximum category ID in dataset1 to ensure new categories from dataset2 are assigned subsequent IDs.
    2. Create a mapping for categories that exist in both datasets to unify their IDs.
    3. For categories in dataset2 that don't exist in dataset1, assign new IDs starting from the last used ID in dataset1 plus one.

'''

import json
import shutil
import os

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_annotations(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def update_image_ids(annotations, start_id):
    image_id_mapping = {image['id']: new_id for new_id, image in enumerate(annotations['images'], start=start_id)}
    for image in annotations['images']:
        image['id'] = image_id_mapping[image['id']]
    for annotation in annotations['annotations']:
        annotation['image_id'] = image_id_mapping[annotation['image_id']]
    return annotations, max(image_id_mapping.values()) + 1

def update_category_ids(annotations, category_mapping):
    """Update the category IDs in the annotations based on the provided mapping."""
    for annotation in annotations['annotations']:
        if annotation['category_id'] in category_mapping:
            annotation['category_id'] = category_mapping[annotation['category_id']]
    return annotations

def unify_categories(data1_categories, data2_categories):
    category_mapping = {}
    new_id_start = max(cat['id'] for cat in data1_categories) + 1
    existing_names = {cat['name']: cat['id'] for cat in data1_categories}
    
    for cat in data2_categories:
        if cat['name'] in existing_names:
            category_mapping[cat['id']] = existing_names[cat['name']]
        else:
            category_mapping[cat['id']] = new_id_start
            data1_categories.append({'id': new_id_start, 'name': cat['name']})
            new_id_start += 1
            
    return category_mapping, data1_categories

def merge_datasets(dataset1_path, dataset2_path, img_dir1, img_dir2, merged_dir):
    data1 = load_annotations(dataset1_path)
    data2 = load_annotations(dataset2_path)

    # Update image IDs in dataset2 to avoid conflicts and ensure unique IDs across the merged dataset
    max_img_id_data1 = max(img['id'] for img in data1['images'])
    data2, next_img_id = update_image_ids(data2, max_img_id_data1 + 1)

    # Unify categories and update category IDs in dataset2 annotations
    category_mapping, unified_categories = unify_categories(data1['categories'], data2['categories'])
    data2 = update_category_ids(data2, category_mapping)

    # Merge images and annotations
    merged_data = {
        "images": data1['images'] + data2['images'],
        "annotations": data1['annotations'] + data2['annotations'],
        "categories": unified_categories
    }

    # Save the merged annotations
    merged_annotations_path = os.path.join(merged_dir, 'merged_annotations.json')
    save_annotations(merged_data, merged_annotations_path)

    # Create the "images" subdirectory within the merged directory
    merged_images_dir = os.path.join(merged_dir, "images")
    if not os.path.exists(merged_images_dir):
        os.makedirs(merged_images_dir)

    # Function to copy images from a source directory to the merged images directory
    def copy_images_to_merged(images, src_dir):
        for img in images:
            src_path = os.path.join(src_dir, img['file_name'])
            dst_path = os.path.join(merged_images_dir, img['file_name'])
            if not os.path.exists(dst_path):  # Avoid overwriting existing files
                shutil.copy(src_path, dst_path)

    # Copy images from both datasets to the merged images directory
    copy_images_to_merged(data1['images'], img_dir1)
    copy_images_to_merged(data2['images'], img_dir2)

    print(f"Merged dataset created with annotations saved to {merged_annotations_path}")
    print(f"Images copied to {merged_images_dir}")

# Example usage
# here, data2 will be merged into data1 (data1 will be the accomulative merge)
merge_datasets(
    '/home/mrt/src/detr/weights/pcc5/person-car/22Feb2025-res50/instances_train.json', # data1 annotations
    '/media/mrt/Whale/data/detr/pcc5/weaver/annotator/fixed_coco.json',  # data2 annotations (will be added to data1)
    '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/train',  # data1 images
    '/media/mrt/Whale/data/detr/pcc5/weaver/images',  # data2 images
    '/media/mrt/Whale/data/detr/pcc5/weaver/New Folder/'# merged pot
)
