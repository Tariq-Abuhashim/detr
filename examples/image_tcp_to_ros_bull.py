
import rclpy
import numpy as np
from snark.imaging.cv_image import iterator, image, make_header_from
import cv2
import sys
from sensor_msgs.msg import NavSatFix, Image
from std_msgs.msg import Header

rclpy.init(args=sys.argv)
node = rclpy.create_node("image_tcp_to_ros")
node.get_logger().info("Launching image_tcp_to_ros")

image_publisher = node.create_publisher(Image, "/bull/camera_image", 10)

if __name__ == '__main__':
    STDIN = sys.stdin.buffer

    
    for i in iterator(STDIN):
        height = int(i.header['rows'])
        width = int(i.header['cols'])
        type = i.header['type']
        total_size = i.data.shape[0]*i.data.shape[1]*i.data.shape[2]
        ros_image = Image()
        image_data = np.array(i.data.reshape(total_size,1).T.flatten(),dtype=np.uint8)
        ros_image.data = bytearray(image_data)
        header = Header()
        header.stamp = node.get_clock().now().to_msg()
        header.frame_id = "/bull/camera_image"
        ros_image.header = header
        ros_image.width = width
        ros_image.height = height
        ros_image.encoding = "rgb8"
        ros_image.is_bigendian = 0
        ros_image.step = width*3
        image_publisher.publish(ros_image)
 
    rclpy.spin(node)
