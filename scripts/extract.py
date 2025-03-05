import cv2
import os

# Path to the video file
video_path = 'videos/man_ir.mp4'

# Output directory where the frames will be saved
output_dir = 'man_ir/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# Setting skip_frames to 10 means every 10th frame will be saved (adjust as needed)
skip_frames = 5
frame_count = 0
saved_frames = 0

while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # Exit the loop if we've reached the end of the video

    #for i in range(3):
    #    cv2.imshow('Channel ' + str(i), frame[:,:,i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #print(frame.type)
    # Check if this frame should be saved
    if frame_count % skip_frames == 0:
        # Construct the output path for the frame
        frame_filename = os.path.join(output_dir, f'frame_{saved_frames:04d}.jpg')

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)
        saved_frames += 1

    frame_count += 1

# Release the video file
cap.release()

print(f"Extracted {saved_frames} frames and saved them to {output_dir}")