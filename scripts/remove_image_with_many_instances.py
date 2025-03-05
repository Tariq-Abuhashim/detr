import json

def remove_images_with_many_objects(input_file, output_file, max_objects=100):
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
    
    # Filter images and annotations
    filtered_images = []
    filtered_annotations = []
    
    for image in coco_data['images']:
        image_id = image['id']
        annotations = image_id_to_annotations.get(image_id, [])
        if len(annotations) <= max_objects:
            filtered_images.append(image)
            filtered_annotations.extend(annotations)
    
    # Update the COCO data with filtered images and annotations
    coco_data['images'] = filtered_images
    coco_data['annotations'] = filtered_annotations
    
    # Save the updated dataset
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"Filtered dataset saved to {output_file}")

# Example usage
input_coco_file = '/media/mrt/Whale/data/detr/VisDrone2019-DET-test-dev/coco_annotations.json'
output_coco_file = '/media/mrt/Whale/data/detr/VisDrone2019-DET-test-dev/100q_coco_annotations.json'
remove_images_with_many_objects(input_coco_file, output_coco_file, max_objects=100)

