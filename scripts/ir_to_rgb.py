
# this takes a folder with ir images, and converts each from 1-channel to 3-channels
# to use it with the 3-channel transformer model

import cv2
import os

# Path to the directory containing your IR images
input_folder = '/home/mrt/dev/detr/data/lmatch/images/'
output_folder = '/home/mrt/dev/detr/data/lmatch/images/tmp'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all images in the directory
for filename in os.listdir(input_folder):
    # Create full path to the image file
    file_path = os.path.join(input_folder, filename)

    # Read the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Ensure it's read as grayscale
    if image is None:
        print(f"Error reading {file_path}. Skipping...")
        continue

    # Check if it's a single channel image
    if len(image.shape) == 2:
        # Convert to 3-channel representation
        image_rgb = cv2.merge([image, image, image])

        # Save the 3-channel image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image_rgb)

print(f"Processed all images and saved them to {output_folder}")