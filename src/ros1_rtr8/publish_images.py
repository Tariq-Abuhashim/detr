import cv2
import os

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError #sudo apt-get install ros-noetic-cv-bridge
from std_msgs.msg import Header

#from detect_trt import TensorRTInference 

def cv2_to_ros_image(cv_image, encoding="bgr8"):
    """
    Convert an OpenCV image to a ROS Image message.
    
    Args:
    - cv_image: OpenCV image to convert
    - encoding: Encoding of the image data ('bgr8', 'mono8', etc.)
    
    Returns:
    - A sensor_msgs/Image message containing the image data.
    """
    # Ensure the image is in the expected format
    if encoding == "bgr8":
        if len(cv_image.shape) == 3:  # color image
            pass  # already in BGR8 format
        elif len(cv_image.shape) == 2:  # grayscale
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)  # convert to BGR
    elif encoding == "mono8" and len(cv_image.shape) == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
    print(cv_image.dtype)
    if len(cv_image.shape) == 3:
        height, width, channels = cv_image.shape
        print("Width:", width)
        print("Channels:", channels)
    else:  # This case handles grayscale images, which don't have a channels dimension
        height, width = cv_image.shape
        print("Width:", width)
        
    #trt_inference = TensorRTInference('../detr.trt')
    #probas, bboxes = trt_inference.detect(cv_image)
    #trt_inference.cleanup() # delete objects to avoid memory segmentation fault
    #print("--- finished ---")

    # Create a ROS Image message
    ros_image = Image()
    ros_image.header = Header(stamp=rospy.Time.now())  # Update with the correct timestamp if necessary
    ros_image.height = cv_image.shape[0]
    ros_image.width = cv_image.shape[1]
    ros_image.encoding = encoding
    ros_image.is_bigendian = False
    ros_image.step = cv_image.shape[1] * cv_image.shape[2] if len(cv_image.shape) == 3 else cv_image.shape[1]
    ros_image.data = cv_image.tostring()

    return ros_image

# Initialize ROS node
rospy.init_node('image_publisher', anonymous=True)

# Create a publisher object
image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)

# Create a CvBridge object
#bridge = CvBridge()

src_directory = '/media/dv/Whale/Orin/annotate/images/'

rate = rospy.Rate(3)  # 1Hz - Adjust the rate as needed

for filename in os.listdir(src_directory):
    if rospy.is_shutdown():
        break
        
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        src_path = os.path.join(src_directory, filename)
        
        # Read the image using OpenCV
        image = cv2.imread(src_path)
        
        # Check if the image was loaded successfully
        if image is not None:
            try:
                # Convert the OpenCV image to a ROS Image message
                ros_image_msg = cv2_to_ros_image(image, "bgr8")
                
                # Publish the ROS Image message
                image_pub.publish(ros_image_msg)
                rospy.loginfo(f"Published {filename}")
                
            except CvBridgeError as e:
                rospy.logerr(f"Could not convert image {filename} to ROS Message. Error: {e}")
                
            rate.sleep()
            
        else:
            rospy.logwarn(f"Failed to load image {filename}")
            
rospy.spin()
