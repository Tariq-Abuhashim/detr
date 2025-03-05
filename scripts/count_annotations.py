import json

# Load annotations
annotations_path = '/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json'

with open(annotations_path) as f:
    annotations = json.load(f)

# Define category names
categories = {0: "unknown", 1: "person", 2: "car", 3: "window"}

# Initialize dictionary to store category counts
category_counts = {}

# Count instances of each category
for annotation in annotations['annotations']:
    category_id = annotation['category_id']
    if category_id in category_counts:
        category_counts[category_id] += 1
    else:
        category_counts[category_id] = 1

# Print category counts
for category_id, count in category_counts.items():
    category_name = categories.get(category_id, "Unknown")
    print(f"{category_name} has {count} examples.")
