from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_image_service',
            executable='image_service',
            name='image_service_node',
            output='screen'
        )
    ])
