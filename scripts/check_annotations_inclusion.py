import json

def load_annotations(file_path):
    """Load the annotations from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def is_bbox_included(smaller_ann, larger_ann):
    """Check if a bounding box from the smaller annotation is included in the larger annotation."""
    for large_ann in larger_ann['annotations']:
        # Check if the image_id and bbox match
        if smaller_ann['image_id'] == large_ann['image_id'] and smaller_ann['bbox'] == large_ann['bbox']:
            return True
    return False

def check_annotations_inclusion(smaller_annotation_file, larger_annotation_file):
    """
    Check if all annotations in the smaller annotation file are included in the larger annotation file.
    """
    smaller_ann = load_annotations(smaller_annotation_file)
    larger_ann = load_annotations(larger_annotation_file)

    included = 0
    not_included = 0

    # Iterate through all annotations in the smaller annotation file
    for ann in smaller_ann['annotations']:
        if is_bbox_included(ann, larger_ann):
            included += 1
        else:
            not_included += 1

    # Print the results
    print(f"Total annotations in smaller file: {len(smaller_ann['annotations'])}")
    print(f"Included annotations: {included}")
    print(f"Not included annotations: {not_included}")

# Example usage
smaller_annotation_file = '/media/mrt/Whale/data/detr/pcc5/army_facebook/via_project_19Feb2025_16h7m_coco.json'
larger_annotation_file = '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/merged_annotations.json'
check_annotations_inclusion(smaller_annotation_file, larger_annotation_file)

