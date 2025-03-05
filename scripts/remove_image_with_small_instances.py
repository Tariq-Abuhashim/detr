import json

def remove_images_with_any_small_boxes(input_file, output_file, min_area):
    """
    Removes images and their annotations if any bounding box has an area smaller than min_area.
    """
    # Load the COCO dataset
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping of image_id to annotations
    image_id_to_annotations = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)
    
    # Initialize counters for debugging
    total_images = len(coco_data['images'])
    removed_images_count = 0

    filtered_images = []
    filtered_annotations = []
    removed_image_ids = set()

    for image in coco_data['images']:
        image_id = image['id']
        annotations = image_id_to_annotations.get(image_id, [])
        
        # Debugging: Check all annotations for the image
        print(f"Processing image_id: {image_id}, annotations: {len(annotations)}")

        # Check for any small bounding boxes
        has_invalid_bbox = False
        for ann in annotations:
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            area = width * height

            # Debugging: Print the bbox and calculated area
            print(f"  Annotation bbox: {ann['bbox']}, Calculated area: {area}")

            if area < min_area:
                has_invalid_bbox = True
                print(f"    -> Image {image_id} flagged for removal due to small bbox: {ann['bbox']} (area: {area})")
                break  # No need to check further annotations for this image

        if not has_invalid_bbox:
            filtered_images.append(image)
            filtered_annotations.extend(annotations)
        else:
            removed_images_count += 1
            removed_image_ids.add(image_id)

    # Remove orphaned annotations
    filtered_annotations = [ann for ann in filtered_annotations if ann['image_id'] not in removed_image_ids]
    
    # Update the COCO data with filtered images and annotations
    coco_data['images'] = filtered_images
    coco_data['annotations'] = filtered_annotations

    # Save the updated dataset
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"\nFiltered dataset saved to {output_file}")
    print(f"Total images: {total_images}, Removed images: {removed_images_count}")

# Example usage
input_coco_file = '/media/mrt/Whale/data/detr/VisDrone2019-DET-test-dev/100q_coco_annotations.json'
output_coco_file = '/media/mrt/Whale/data/detr/VisDrone2019-DET-test-dev/475p^2_100q_coco_annotations.json'
remove_images_with_any_small_boxes(input_coco_file, output_coco_file, 475)

