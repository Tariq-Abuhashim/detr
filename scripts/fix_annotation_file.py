# This file takes the exported COCO format annotations from VGG image annotator
# it removes duplicate annotations, and tidies up the file
# it is easier to open using mousepad or gedit after.

import json

# Load the COCO formatted JSON file
with open('/media/mrt/Whale/data/detr/pcc5/weaver/annotator/via_project_26Feb2025_0h16m_coco.json', 'r') as f:
    coco_data = json.load(f)

# Process annotations to remove duplicates
unique_annotations = []
seen = set()

for annotation in coco_data['annotations']:
    # Using a tuple of important keys to determine if we've seen this annotation before
    identifier = (annotation['image_id'], tuple(annotation['bbox']), annotation['category_id'])
    if identifier not in seen:
        seen.add(identifier)
        unique_annotations.append(annotation)

# Overwrite annotations with the unique set
coco_data['annotations'] = unique_annotations

# Save the cleaned COCO formatted JSON file
with open('/media/mrt/Whale/data/detr/pcc5/weaver/annotator/fixed_coco.json', 'w') as f:
    json.dump(coco_data, f, indent=4)
