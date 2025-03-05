import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from infer_engine import TensorRTInference
import numpy as np

class TriggerAndListenNode(Node):

    def __init__(self):
        super().__init__('trigger_and_listen_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/vulcan1/camera_image', self.image_callback, 10)

        self.detector = TensorRTInference('model.engine')
        self.detection_pub = self.create_publisher(Detection2DArray, '/vulcan1/detection_results', 10)

        self.trigger_image_service()
        
    def trigger_image_service(self):
        self.get_logger().info('Attempting to trigger image capture...')
        self.client = self.create_client(Trigger, 'trigger_image')
        # while not self.client.wait_for_service(timeout_sec=1.0):
            # self.get_logger().info('Service not available, waiting again...')

        self.req = Trigger.Request()
        self.future = self.client.call_async(self.req)
        self.future.add_done_callback(self.trigger_response_callback)

    def trigger_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Trigger image service called successfully.')
            else:
                self.get_logger().error('Failed to call trigger image service.')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def image_callback(self, msg):
        self.get_logger().info('Image received.')
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        # print(cv_image)
        try:
            height, width, channels = cv_image.shape
            self.get_logger().info(f'Image size: {width}x{height}, Channels: {channels}')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            probas, bboxes = self.detector.infer(cv_image)
            self.get_logger().info(f'Number of detections: {len(bboxes)}')

            detection_array_msg = Detection2DArray()

            for prob, bbox in zip(probas, bboxes):
                
                detection_msg = Detection2D()

                result = ObjectHypothesisWithPose()
                result.hypothesis.class_id = str(self.detector.CLASSES[np.argmax(prob)])
                result.hypothesis.score = float(np.max(prob))
                detection_msg.results.append(result)

                bbox_msg = detection_msg.bbox
                bbox_msg.center.position.x = (bbox[0] + bbox[2]) / 2.0
                bbox_msg.center.position.y = (bbox[1] + bbox[3]) / 2.0
                bbox_msg.size_x = float(bbox[2] - bbox[0])
                bbox_msg.size_y = float(bbox[3] - bbox[1])

                detection_array_msg.detections.append(detection_msg)
            
            # if len(probas) > 0:
            self.detection_pub.publish(detection_array_msg)
            self.get_logger().info('Detection results published...')
            
            self.trigger_image_service()  # Trigger the next image

        except Exception as e:
            self.get_logger().error(f'Failed to display image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TriggerAndListenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
