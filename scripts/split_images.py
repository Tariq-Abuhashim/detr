import os
import shutil
import json
from sklearn.model_selection import train_test_split

# Paths

annotation_file = '/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json'
images_folder =   '/media/mrt/Whale/data/detr/pcc5/person-car/all_images'

train_folder =    '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/train'
val_folder =      '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/val'
test_folder =     '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/test'
anno_folder =     '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/annotations'

# Ensure target folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(anno_folder, exist_ok=True)

# Load annotations
with open(annotation_file, 'r') as f:
    data = json.load(f)

# Get unique image filenames from annotations
image_filenames = {item['file_name'] for item in data['images']}

# First, split the data into training (80%) and temporary set (20%).
train_filenames, temp_filenames = train_test_split(
    list(image_filenames), 
    test_size=0.075, 
    random_state=42  # For reproducibility
)

# Now, split the temporary set into validation (50%) and test (50%) sets. 
# This means we're splitting the original data into 80% train, 10% val, and 10% test.
val_filenames, test_filenames = train_test_split(
    temp_filenames, 
    test_size=0.2, # was 0.5 
    random_state=42  # For reproducibility
)

# Copy images to respective folders
for filename in train_filenames:
    shutil.copy(os.path.join(images_folder, filename),
                os.path.join(train_folder, filename))

for filename in val_filenames:
    shutil.copy(os.path.join(images_folder, filename),
                os.path.join(val_folder, filename))

for filename in test_filenames:
    shutil.copy(os.path.join(images_folder, filename),
                os.path.join(test_folder, filename))

print(f"Copied {len(train_filenames)} images to {train_folder}")
print(f"Copied {len(val_filenames)} images to {val_folder}")
print(f"Copied {len(test_filenames)} images to {test_folder}")

# If you wish to split annotations as well, you can further process the 'data' object.

# Extract IDs of train and val images
train_ids = [img['id'] for img in data['images'] if img['file_name'] in train_filenames]
val_ids = [img['id'] for img in data['images'] if img['file_name'] in val_filenames]
test_ids = [img['id'] for img in data['images'] if img['file_name'] in test_filenames]

# Filter the annotations and images for train and val sets
train_annotations = [anno for anno in data['annotations'] if anno['image_id'] in train_ids]
val_annotations = [anno for anno in data['annotations'] if anno['image_id'] in val_ids]
test_annotations = [anno for anno in data['annotations'] if anno['image_id'] in test_ids]

train_images = [img for img in data['images'] if img['id'] in train_ids]
val_images = [img for img in data['images'] if img['id'] in val_ids]
test_images = [img for img in data['images'] if img['id'] in test_ids]

# Create new dictionaries for train and val annotations
train_data = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": data['categories']
}

val_data = {
    "images": val_images,
    "annotations": val_annotations,
    "categories": data['categories']
}

test_data = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": data['categories']
}

# Save the split annotation files
with open(os.path.join(anno_folder, 'instances_train.json'), 'w') as f:
    json.dump(train_data, f, indent=4)

with open(os.path.join(anno_folder, 'instances_val.json'), 'w') as f:
    json.dump(val_data, f, indent=4)

with open(os.path.join(anno_folder, 'instances_test.json'), 'w') as f:
    json.dump(test_data, f, indent=4)

print(f"Saved train annotations to instances_train.json")
print(f"Saved val annotations to instances_val.json")
print(f"Saved test annotations to instances_test.json")

