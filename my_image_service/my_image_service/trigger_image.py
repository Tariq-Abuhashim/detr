import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class TriggerAndListenNode(Node):

    def __init__(self):
        super().__init__('trigger_and_listen_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/arena_camera_node/resized/images', self.image_callback, 10)
        self.trigger_image_service()

    def trigger_image_service(self):
        self.get_logger().info('Attempting to trigger image capture...')
        self.client = self.create_client(Trigger, 'trigger_image')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            height, width, channels = cv_image.shape
            self.get_logger().info(f'Image size: {width}x{height}, Channels: {channels}')
            #cv2.imshow('Received Image', cv_image)
            #cv2.waitKey(1)
            #cv2.destroyAllWindows()
            self.trigger_image_service()  # Trigger the next image after displaying the current one
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
