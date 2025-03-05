#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# from detection_msgs.msg import BoundingBox, DetectionResults
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

from infer_engine import TensorRTInference

def ros_image_to_cv2(ros_image, encoding="bgr8"):
    """
    Convert a ROS image message to an OpenCV image without using cv_bridge.

    Args:
    - ros_image: The ROS image message to convert.
    - encoding: The desired encoding for the OpenCV image. Defaults to "bgr8".

    Returns:
    - cv_image: A NumPy array representing the OpenCV image.
    """
    if encoding not in ["bgr8", "mono8"]:
        raise ValueError("Unsupported encoding: {}".format(encoding))
    
    dtype = np.uint8
    n_channels = 3 if encoding == "bgr8" else 1
    
    # Decode the image data and reshape it based on encoding
    cv_image = np.frombuffer(ros_image.data, dtype=dtype).reshape(ros_image.height, ros_image.width, n_channels)
    
    if encoding == "mono8":
        # If the image is grayscale, it's already in the correct shape
        pass
    elif encoding == "bgr8":
        # If the image is color, no further action required for bgr8,
        # but if you want to convert it to RGB, you can uncomment the following line: 
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pass
    
    return cv_image

class ObjectDetector( Node ):
    def __init__(self):
        self.node_name = "object_detector"

        self.detector = TensorRTInference('model.engine')

        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback, queue_size=1)
        # self.detection_pub = rospy.Publisher("/detection_results", DetectionResults, queue_size=10)

        super().__init__('object_detector')
        self.subscription = self.create_subscription( Image, '/camera/image_raw', self.callback, 10)
        # self.detection_pub = rospy.Publisher("/detection_results", DetectionResults, queue_size=10)
        # self.subscription  # prevent unused variable warning
        self.br = CvBridge()
 
    def callback(self, msg):
        print("I got a message")
        try:
            cv_image = ros_image_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        print(cv_image.dtype)
        if len(cv_image.shape) == 3:
            height, width, channels = cv_image.shape
            print("Width:", width)
            print("Channels:", channels)
            print("Height:", height)
        else:  # This case handles grayscale images, which don't have a channels dimension
            height, width = cv_image.shape
            print("Width:", width)


        # if required, resize the image to the expected size
        # expected size is: 800, 1422
        if cv_image.shape[0] != 800 or cv_image.shape[1] != 1422:
            cv_image = cv2.resize(cv_image, (1422, 800))
            print("Resized image to 800x1422")

        # Run detection
        #cv_image = cv2.imread('/media/dv/Whale/Orin/annotate/images/test/20230210T081116.774521.png')
        probas, bboxes = self.detector.infer(cv_image)
        print("Probas:", probas)
        print("Bboxes:", bboxes)

def main(args=None):
    rclpy.init(args=args)
    object_detector = ObjectDetector()
    rclpy.spin(object_detector)
    object_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()