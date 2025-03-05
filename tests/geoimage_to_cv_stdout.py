import rclpy
import sys
from snark.imaging.cv_image import image, make_header_from
from cv_bridge import CvBridge
import copy
import cv2
from ms_civtak_ros_bridge_msgs.msg import GeoImage

class GeoImageToCV():

    def __init__(self, node):
        self.node = node

        self.geo_image_subscriber = self.node.create_subscription(GeoImage, "/vulcan1/geo_image", self.send_geoimage, 10)
   
    def send_geoimage(self, msg):

        assert isinstance(msg, GeoImage)

        geoimage = msg
        img = geoimage.image

        pose = geoimage.position

        ros_cv_bridge = CvBridge()


        cv_image = ros_cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

        # cv_image =  cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        image(make_header_from(cv_image),cv_image).write(flush=False)



if __name__ == "__main__":

    rclpy.init(args=sys.argv)
    node = rclpy.create_node("geoimage_to_cv")
    node.get_logger().info("Launching geoimage to cv")
    
    ros_cv = GeoImageToCV(node)
    
    rclpy.spin(node)

    
