cmake_minimum_required(VERSION 3.5)
project(my_image_service)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    my_image_service/image_service.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()