# this is a sample file to show how to create a coco format annotation file
# It has not been used yet

data = {
    "images": [],
    "annotations": [],
    "categories": []
}

category_map = {"gun": 1, "knife": 2}  # Modify as per your categories
data["categories"] = [{"id": v, "name": k} for k, v in category_map.items()]

for idx, frame in enumerate(frames):
    # Add image info
    data["images"].append({
        "id": idx,
        "width": frame_width,
        "height": frame_height,
        "file_name": f"frame_{idx:04d}.jpg"
    })

    # Add annotations (You would fetch these from your annotation tool's data)
    for ann in frame_annotations:
        data["annotations"].append({
            "id": annotation_id, 
            "image_id": idx,
            "bbox": [ann.x, ann.y, ann.width, ann.height],
            "category_id": category_map[ann.label]
        })
        annotation_id += 1

# Save to a JSON file
import json
with open('coco_format_annotations.json', 'w') as f:
    json.dump(data, f)