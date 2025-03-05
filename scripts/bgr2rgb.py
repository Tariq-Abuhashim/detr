import cv2
import os
from pathlib import Path

# Define the source and destination directories
src_directory = '/home/mrt/dev/detr/data/ms/person+car/original/images/'
dst_directory = '/home/mrt/dev/detr/data/ms/person+car/original/images//'

# Create the destination directory if it doesn't exist
Path(dst_directory).mkdir(parents=True, exist_ok=True)

# Iterate over all files in the source directory
for filename in os.listdir(src_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  # Add/check other file extensions if necessary
        # Construct the full file path
        src_path = os.path.join(src_directory, filename)
        
        # Read the image in BGR format
        bgr_image = cv2.imread(src_path)
        
        # Check if the image was successfully loaded
        if bgr_image is not None:
            # Convert the image from BGR to RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Construct the full path for the destination
            dst_path = os.path.join(dst_directory, filename)
            
            # Save the converted image
            # Note: cv2.imwrite saves images in BGR format, so if you're saving the image to view in external programs,
            # you might want to convert it back to BGR or use another library like imageio or PIL to save in RGB format.
            cv2.imwrite(dst_path, rgb_image)
            
            print(f"Converted and saved {filename}")
        else:
            print(f"Failed to load {filename}")
    else:
        print(f"Skipped {filename}, unsupported file type.")

print("Conversion completed.")
