import cv2
import json
import os

# Directory containing the text files
folder_path = '/media/mrt/Whale/data/detr/VisDrone2019-DET-test-dev/'  # applicable for this data format 
output_file = '/media/mrt/Whale/data/detr/VisDrone2019-DET-test-dev/coco_annotations.json'

# List all files in the directory
file_names = os.listdir(folder_path+"/annotations")

# Initialize COCO-format dictionary with categories
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 0, "name": "unknown"},  # Assuming '0' is used for unknown
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"}#,
        #{"id": 3, "name": "window"},
    ]
}

# Mapping from label names to category IDs
label_to_category_id = {
    1: 1, #pedestrian(1) to person(1)
    2: 1, #people(2) to person(1)
    4: 2  #car(4) to car(s)
}

# Trackers for unique IDs
image_id_tracker = 1
annotation_id_tracker = 1
image_ids = {}

# Iterate over each file
for file_name in file_names:
    # Check if the file is a text file
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path+"/annotations", file_name)
        
        # Read image
        image_name = file_name[0:-3] + "jpg"
        image_path = os.path.join(folder_path+"/images", image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            continue
        coco_format["images"].append({
            "file_name": image_name,
            "id": image_id_tracker,
            "width": 0,  # Placeholder, update if width is known
            "height": 0  # Placeholder, update if height is known
        })

        # Open the file and read its contents
        with open(file_path, 'r') as file:
            #content = file.read()
            #print(f"Contents of {file_name}:")
            #print(content)
            #for line in file:
            #    print(line.strip())
            for line_number, line in enumerate(file, start=1):
                values = line.split(',')
                numbers = [int(x.strip()) for x in values if x.strip()]
                #print(f"Numbers in line {line_number}: {numbers}")
                x,y,w,h,score,object_category,truncation,occlusion = numbers
                #print("--------------------------------------")
                if object_category in label_to_category_id:
                    coco_format["annotations"].append({
                        "id": annotation_id_tracker,
                        "image_id": image_id_tracker,
                        "category_id": label_to_category_id[object_category],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    annotation_id_tracker+=1

        image_id_tracker+=1

# Save the COCO-format data
with open(output_file, 'w') as f:
    json.dump(coco_format, f, indent=4)
            
