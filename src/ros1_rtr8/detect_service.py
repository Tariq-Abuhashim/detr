#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBox, DetectionResults
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

from detect_trt import TensorRTInference 
from std_srvs.srv import Trigger, TriggerRequest  # Import the Trigger service

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

class ObjectDetector:
    def __init__(self):
        self.node_name = "object_detector"
        rospy.init_node(self.node_name, anonymous=True)

        self.bridge = CvBridge()
        self.detection_pub = rospy.Publisher("/detection_results", DetectionResults, queue_size=10)

        rospy.wait_for_service('camera_service')
        self.camera_service_client = rospy.ServiceProxy('camera_service', Trigger)

        self.detector = TensorRTInference(engine_path="detr.trt")

    def get_image_from_service(self):
        try:
            sos = TriggerRequest()
            response = self.camera_service_client(sos)
            if response.success:
                return response.image  # Assuming the service response has an image field
            else:
                rospy.logerr("Service call failed: %s" % response.message)
                return None
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None

    def detect_objects(self, msg):
        ros_image = self.get_image_from_service()
        if ros_image is None:
            return

        try:
            #cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
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
        else:
            height, width = cv_image.shape
            print("Width:", width)

        # Run detection
        #cv_image = cv2.imread('/media/dv/Whale/Orin/annotate/images/test/20230210T081116.774521.png')
   #     probas, bboxes_scaled = self.detector.detect(cv_image)

        # Prepare DetectionResults message
   #     detection_msg = DetectionResults()
   #     detection_msg.bounding_boxes = []

   #     for prob, bbox in zip(probas, bboxes_scaled):
   #         bbox_msg = BoundingBox()
   #         bbox_msg.x_min, bbox_msg.y_min, bbox_msg.x_max, bbox_msg.y_max = bbox
   #         bbox_msg.confidence = float(np.max(prob))
   #         bbox_msg.class_label = str(self.detector.CLASSES[np.argmax(prob)])
   #         detection_msg.bounding_boxes.append(bbox_msg)

   #     self.detection_pub.publish(detection_msg)
        
   #     rospy.loginfo("Published bounding boxes")

if __name__ == '__main__':
    detector = ObjectDetector()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        detector.detect_objects()
        rate.sleep()

