#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import os

class ImageServiceNode(Node):

    def __init__(self):
        super().__init__('image_service_node')
        self.bridge = CvBridge()
        self.create_service(Trigger, 'trigger_image', self.handle_trigger_image)
        self.publisher = self.create_publisher(Image, '/arena_camera_node/resized/images', 10)
        self.image_path = '/ros2_ws/src/my_image_service/frame.0000.color.png'
        self.get_logger().info('Image Service Node has been started.')

    def handle_trigger_image(self, request, response):
        self.get_logger().info('Trigger image service called.')

        # Read the image from disk
        if os.path.exists(self.image_path):
            cv_image = cv2.imread(self.image_path)
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            self.publisher.publish(ros_image)
            response.success = True
            response.message = 'Image published successfully.'
            self.get_logger().info('Image published.')
        else:
            response.success = False
            response.message = 'Image file not found.'
            self.get_logger().error('Image file not found.')

        return response

def main(args=None):
    rclpy.init(args=args)
    node = ImageServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

