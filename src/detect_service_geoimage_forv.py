import rclpy
import argparse
import sys
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from threading import Thread, Lock
from geopy.distance import geodesic

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
 
qos_profile = QoSProfile(
	reliability=QoSReliabilityPolicy.BEST_EFFORT,
	history=QoSHistoryPolicy.KEEP_LAST,
	depth=10
)

# from snark.imaging.cv_image import image, make_header_from

from sensor_msgs.msg import NavSatFix, Image
from geometry_msgs.msg import PoseStamped, PoseWithCovariance
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geographic_msgs.msg import GeoPoseStamped
from infer_engine import TensorRTInference
from ms_civtak_ros_bridge_msgs.msg import GeoImage
import copy
import numpy as np

class TriggerAndListenNode(Node):

    def __init__(self, engine='model.engine', num_classes=3, namespace="vulcan1", trained_width=1024, trained_height=800, filter_time=5, filter_spatial=5):
        super().__init__('detection_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, f'/arena_camera_node/images', self.image_callback, 10)

        self.detector = TensorRTInference(engine, num_classes)
        # self.detection_pub = self.create_publisher(Detection2DArray, '/vulcan1/detection_results', 10)

        self.last_geo_pose = NavSatFix()
        self.last_geo_pose_lock = Lock()
        
        self.last_pose = PoseStamped()
        self.last_pose_lock = Lock()

        self.last_image = Image()
        self.last_image_lock = Lock()
        self.ros_cv_bridge = CvBridge()

        self.last_geo_image = GeoImage()
        self.last_geo_image_lock = Lock()

        self.index = 0

        self.trained_width = trained_width
        self.trained_height = trained_height

        self.filter_time = filter_time
        self.filter_spatial = filter_spatial

        self.get_logger().info('filter_time: ' + str(self.filter_time))
        self.get_logger().info('filter_space: ' + str(self.filter_spatial))

        self.geo_image_publisher = self.create_publisher(GeoImage, f"/{namespace}/civtak_ros_bridge/geo_image", 10)
        self.object_hypothesis_with_pose_publisher = self.create_publisher(ObjectHypothesisWithPose, f"/{namespace}/detection_node/detection_results/threats", 10)

        self.pose_subscriber_navsat = self.create_subscription(NavSatFix, "/an_device/NavSatFix", self.update_geopose, 10)
        # self.pose_subscriber_geopose = self.create_subscription(GeoPoseStamped, f"/{namespace}/geo_pose", self.update_geopose, 10)
        self.pose_subscriber_pose = self.create_subscription(PoseStamped, f"/{namespace}/pose", self.update_pose, qos_profile)


        # self.trigger_image_service()

    def update_geopose(self, msg):
        with self.last_geo_pose_lock:
            self.last_geo_pose = msg

        
    def update_pose(self, msg):
        with self.last_pose_lock:
            self.last_pose = msg

    def update_image(self, msg):
        with self.last_image_lock:
            self.last_image = msg
        
    def get_image(self):
        with self.last_image_lock:
            image = copy.copy(self.last_image)
        return image

    def get_geopose(self):
        with self.last_geo_pose_lock:
            geo_pose = copy.copy(self.last_geo_pose)
        return geo_pose
    
    def get_pose(self):
        with self.last_pose_lock:
            pose = copy.copy(self.last_pose)
        return pose
    
    def update_last_detection_geo_image(self, msg):
        with self.last_geo_image_lock:
            self.last_geo_image = msg

    def get_last_detection_geo_image(self):
        with self.last_geo_image_lock:
            geo_image = copy.copy(self.last_geo_image)
        return geo_image
    
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
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # print(cv_image)
        try:
            height, width, channels = cv_image.shape 
            
            
            half_trained_width = int(np.round(self.trained_width/2))
            half_trained_height = int(np.round(self.trained_height/2))

            centr_point_x = int(np.round(width/2))
            # center_point_y = height - int(np.round(height/2))
            center_point_y = height - int(np.round(self.trained_height/2))

            self.get_logger().info(f'Image size: {width}x{height}, Channels: {channels}')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            cv_image = cv_image[center_point_y - half_trained_height:center_point_y+half_trained_height, centr_point_x-half_trained_width:centr_point_x+half_trained_width, :] # crop cv_image
            
            # cv_image =  cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


            probas, bboxes = self.detector.infer(cv_image)
            self.get_logger().info(f'Number of detections: {len(bboxes)}')

            if len(bboxes):
                self.index +=1

            detection_array_msg = Detection2DArray()

            pose = self.get_pose()

            geo_pose = self.get_geopose()

            last_geo_image = self.get_last_detection_geo_image()

            if len(last_geo_image.image.data):
                timestamp = geo_pose.header.stamp
                new_image_coord = (geo_pose.latitude, geo_pose.longitude)
                # new_image_coord = (50*np.random.random()-25, 50*np.random.random()-25)
                should_filter_time = (np.abs(timestamp.sec - last_geo_image.position.header.stamp.sec) < self.filter_time and self.filter_time > 0)
                should_filter_space = (geodesic(new_image_coord, (last_geo_image.position.latitude,last_geo_image.position.longitude)).meters < self.filter_spatial and self.filter_spatial > 0)
                if timestamp.sec < last_geo_image.position.header.stamp.sec or should_filter_time or should_filter_space:
                        return


            for prob, bbox in zip(probas, bboxes):
                
                detection_msg = Detection2D()

                result = ObjectHypothesisWithPose()
                result.hypothesis.class_id = str(self.detector.CLASSES[np.argmax(prob)])

                if result.hypothesis.class_id != "person": # skip if not a person
                    continue

                result.hypothesis.score = float(np.max(prob))
                pose_with_covariance = PoseWithCovariance()
                pose_with_covariance.pose = pose.pose
                result.pose = pose_with_covariance
                detection_msg.results.append(result)

                self.object_hypothesis_with_pose_publisher.publish(result)

                bbox_msg = detection_msg.bbox
                bbox_msg.center.position.x = (bbox[0] + bbox[2]) / 2.0
                bbox_msg.center.position.y = (bbox[1] + bbox[3]) / 2.0
                bbox_msg.size_x = float(bbox[2] - bbox[0])
                bbox_msg.size_y = float(bbox[3] - bbox[1])

                detection_array_msg.detections.append(detection_msg)
            
            if len(probas) == 0:
               return
            

            detections = detection_array_msg
            #self.get_logger().info(str(detections.results))
            # if not detections.detections:
            #     return

            image_ros = self.get_image()

            # self.node.get_logger().info(image_ros)
            #cv_image = self.ros_cv_bridge.imgmsg_to_cv2(image_ros, encoding="rgb8")
            
            
            for detection in detections.detections:
                bbox = detection.bbox
                center = bbox.center
                size_x = bbox.size_x
                size_y = bbox.size_y
                # self.node.get_logger().info(center)
                top_left_x = center.position.x - size_x
                top_left_y = center.position.y + size_y
            
                bottom_right_x = center.position.x + size_x
                bottom_right_y = center.position.y - size_y

                class_id = detection.results[0].hypothesis.class_id
                probability = str(int(np.round(detection.results[0].hypothesis.score*100))) + "%"
                self.get_logger().info(str(detection.results[0].hypothesis))

                self.get_logger().info(str(probability))

                # print(probability)
                colour = (255,255,255) # black in bgr
                if class_id == "person":
                    # colouring red (bgr)
                    colour = (0,0,255)
                elif class_id == "window":
                    # colouring green (bgr)
                    colour = (0,255,0)
                elif class_id == "car":
                    # colouring blue (bgr)
                    colour = (255,0,0)
                
                cv_image = cv2.rectangle(cv_image, (int(top_left_x),int(top_left_y)),(int(bottom_right_x), int(bottom_right_y)), colour, 5)


                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (int(top_left_x),int(bottom_right_y)-40)
                fontScale = 1
                thickness = 2
                cv2.putText(cv_image, class_id.capitalize(), org, font,  
                   fontScale, colour, thickness, cv2.LINE_AA) 
                
                org = (int(top_left_x),int(bottom_right_y)-10)

                cv2.putText(cv_image, probability, org, font,  
                   fontScale, colour, thickness, cv2.LINE_AA) 
            self.get_logger().info("publishing detection")

            detection_image = GeoImage()
            ros_cv_bridge = CvBridge()

            cv_image =  cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            detection_image.position.latitude = geo_pose.latitude
            detection_image.position.longitude = geo_pose.longitude
            detection_image.position.header = geo_pose.header

            # image(make_header_from(cv_image),cv_image).write(flush=False)

            detection_image.image = ros_cv_bridge.cv2_to_imgmsg(cv_image, encoding="rgb8", header=image_ros.header)

            detection_image.icon_title.data=f"Detection " + str(self.index)
            
            self.geo_image_publisher.publish(detection_image)

            self.update_last_detection_geo_image(detection_image)

            

            self.get_logger().info('Detection results published...')
            
            #self.trigger_image_service()  # Trigger the next image

        except Exception as e:
            self.get_logger().error(f'Failed to display image: {e}')

def main(args=None):
    parser = argparse.ArgumentParser(description='Predict and recreate images with missing red channel.')
    parser.add_argument('--engine', type=str, default='model.engine', help='Model engine name')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--n', type=str, default=3, help='Namespace')
    parser.add_argument('--trained_width', type=int, default=1024, help='Width that detection model was trained on.')
    parser.add_argument('--trained_height', type=int, default=800, help='Height that detection model was trained on.')
    parser.add_argument('--time_filter', type=float, default=0, help='Detection filtered by seconds.')
    parser.add_argument('--spatial_filter', type=float, default=0, help='Detection filtered by meters.')


    parsed_args, unknown = parser.parse_known_args()

    # Filter out custom arguments from sys.argv so that rclpy.init doesn't process them
    sys.argv = [sys.argv[0]] + unknown

    rclpy.init(args=sys.argv)
    node = TriggerAndListenNode(engine=parsed_args.engine, num_classes=parsed_args.num_classes, namespace=parsed_args.n, trained_width=parsed_args.trained_width, trained_height=parsed_args.trained_height, filter_time=parsed_args.time_filter, filter_spatial=parsed_args.spatial_filter)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
