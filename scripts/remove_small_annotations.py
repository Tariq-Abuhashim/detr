import json

# Load the COCO annotations JSON file
with open('/home/mrt/dev/detr/data/VisDrone2019-DET-train/coco_annotations.json') as f:
    data = json.load(f)

# Function to check if the annotation is smaller than the specified size
def is_smaller_than(annotation, min_width, min_height):
    # COCO bounding boxes are formatted as [x, y, width, height]
    _, _, width, height = annotation['bbox']
    return width < min_width or height < min_height

# Minimum dimensions
min_width = 15
min_height = 30

# Filter out annotations smaller than the specified size
filtered_annotations = [ann for ann in data['annotations'] if not is_smaller_than(ann, min_width, min_height)]

# Update the annotations in the original data
data['annotations'] = filtered_annotations

# Save the updated data to a new JSON file
with open('/home/mrt/dev/detr/data/VisDrone2019-DET-train/larger_than_15_30.json', 'w') as f:
    json.dump(data, f)

print(f"Remaining annotations: {len(filtered_annotations)}")
