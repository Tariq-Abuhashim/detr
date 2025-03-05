import json
import csv

# Paths to your files
annotations_file = '/home/mrt/Downloads/via_project_17Feb2024_11h59m_csv.csv'  # applicable for this data format
output_file = '/home/mrt/Downloads/coco_annotations.json'

# Initialize COCO-format dictionary with categories
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "window"},
        {"id": 0, "name": "unknown"}  # Assuming '0' is used for unknown
    ]
}

# Mapping from label names to category IDs
label_to_category_id = {
    "person": 1,
    "car": 2,
    "window": 3,
}

# Trackers for unique IDs
image_id_tracker = 1
annotation_id_tracker = 1
image_ids = {}

# Process annotations
with open(annotations_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            file_name, _, _, _, _, bbox_str, label_str = row
            bbox = json.loads(bbox_str.replace("'", "\""))  # Ensure proper JSON format
            label_data = json.loads(label_str.replace("'", "\""))  # Ensure proper JSON format
            label_name = label_data.get("name", "").lower()  # Extract label name and convert to lowercase
            category_id = label_to_category_id.get(label_name, 0)  # Map to category ID, default to 0 (unknown)

            if file_name not in image_ids:
                image_ids[file_name] = image_id_tracker
                coco_format["images"].append({
                    "file_name": file_name,
                    "id": image_id_tracker,
                    "width": 0,  # Placeholder, update if width is known
                    "height": 0  # Placeholder, update if height is known
                })
                image_id_tracker += 1

            coco_format["annotations"].append({
                "id": annotation_id_tracker,
                "image_id": image_ids[file_name],
                "category_id": category_id,
                "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                "area": bbox['width'] * bbox['height'],
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id_tracker += 1

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row: {row}, error: {e}")
        except KeyError as e:
            print(f"Missing key in row: {row}, error: {e}")

# Save the COCO-format data
with open(output_file, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"Converted annotations saved to {output_file}")
