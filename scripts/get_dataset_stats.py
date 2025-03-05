import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_coco_annotations(annotation_path):
    """
    Loads COCO annotations from the JSON file.
    """
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def calculate_statistics(coco_data):
    """
    Prints statistics about COCO annotations: image count, class distribution, box dimensions, etc.
    """
    # Number of images
    num_images = len(coco_data['images'])
    print(f"Number of images: {num_images}")
    
    # Number of annotations
    num_annotations = len(coco_data['annotations'])
    print(f"Number of annotations: {num_annotations}")
    
    # Number of classes
    categories = coco_data['categories']
    category_names = [category['name'] for category in categories]
    num_classes = len(categories)
    print(f"Number of classes: {num_classes}")
    
    # Class distribution
    category_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        category_counts[ann['category_id']] += 1
    
    print("\nClass distribution (category_id -> count):")
    for category in category_counts:
        category_name = category_names[category]
        print(f"{category_name}: {category_counts[category]}")
    
    # Bounding box dimension statistics
    bbox_widths = []
    bbox_heights = []
    for ann in coco_data['annotations']:
        x_min, y_min, w, h = ann['bbox']
        bbox_widths.append(w)
        bbox_heights.append(h)
    
    print("\nBounding box dimension statistics:")
    print(f"Width: min={min(bbox_widths)}, max={max(bbox_widths)}, mean={np.mean(bbox_widths)}, median={np.median(bbox_widths)}")
    print(f"Height: min={min(bbox_heights)}, max={max(bbox_heights)}, mean={np.mean(bbox_heights)}, median={np.median(bbox_heights)}")
    
    # Distribution of bbox width and height
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(bbox_widths, bins=50, color='blue', alpha=0.7)
    plt.title("Bounding Box Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(bbox_heights, bins=50, color='red', alpha=0.7)
    plt.title("Bounding Box Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

    # Annotations per image
    image_annotation_count = defaultdict(int)
    for ann in coco_data['annotations']:
        image_annotation_count[ann['image_id']] += 1
    
    # Plot annotations per image
    num_annotations_per_image = list(image_annotation_count.values())
    plt.figure(figsize=(6, 4))
    plt.hist(num_annotations_per_image, bins=range(1, max(num_annotations_per_image) + 1), color='green', alpha=0.7)
    plt.title("Annotations per Image")
    plt.xlabel("Number of Annotations")
    plt.ylabel("Frequency")
    plt.show()

# Path to your COCO annotations file
annotation_path = '/media/mrt/Whale/data/detr/pcc5/person-car/03-02-2025/merged_annotations.json'
coco_data = load_coco_annotations(annotation_path)
calculate_statistics(coco_data)

