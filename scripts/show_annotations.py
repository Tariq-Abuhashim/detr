import cv2
import json
import os

# Load annotations
annotations_path = '/media/mrt/Whale/data/detr/pcc5/weaver/annotator/via_project_25Feb2025_21h9m_coco.json'
images_folder = '/media/mrt/Whale/data/detr/pcc5/weaver/images'

with open(annotations_path) as f:
    annotations = json.load(f)

categories = { 1:"person", 2:"car", 3:"window", 4:"unknown",  }

# Function to draw bounding boxes on image
def draw_bounding_boxes(image, annotations):
    for ann in annotations:
        # Assuming the format [x, y, width, height]
        bbox = ann['bbox']
        x, y, w, h = bbox
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)  # Blue box with thickness 2

        # Optionally, add category name if 'categories' dict is provided
        if categories and ann["category_id"] in categories:
            category_name = categories[ann["category_id"]]
            text = f"{ann['category_id']}: {category_name}"
            cv2.putText(image, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Iterate through each image in annotations
for image_info in annotations['images']:
    image_id = image_info['id']
    image_path = os.path.join(images_folder, image_info['file_name'])

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        continue

    # Filter annotations for the current image
    image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

    # Draw bounding boxes on the image
    draw_bounding_boxes(image, image_annotations)

    # Dynamically resize the image based on a desired width
    desired_width = 800
    scale_ratio = desired_width / image.shape[1]
    resized_image_dynamic = cv2.resize(image, (int(image.shape[1] * scale_ratio), int(image.shape[0] * scale_ratio)))

    # Display the resized image
    cv2.imshow("Resized Image", resized_image_dynamic)
    key = cv2.waitKey(0)  # Wait indefinitely for a key press
    if key == ord('q'):  # Press 'q' to quit early
        break

cv2.destroyAllWindows()
