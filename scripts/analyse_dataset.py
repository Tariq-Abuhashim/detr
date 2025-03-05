import json
from collections import defaultdict

def analyze_dataset(annotation_file):
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    
    size_distribution = defaultdict(int)
    
    for annotation in dataset['annotations']:
        bbox = annotation['bbox']
        area = bbox[2] * bbox[3]  # width * height
        
        if area < 32**2:
            size_distribution['small'] += 1
        elif area < 96**2:
            size_distribution['medium'] += 1
        else:
            size_distribution['large'] += 1
    
    return size_distribution

# Path to your dataset's annotation file
annotation_file = '/media/mrt/Whale/data/detr/pcc5/person-car/all_annotations.json'
size_distribution = analyze_dataset(annotation_file)
print("Size Distribution:", size_distribution)
