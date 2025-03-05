# Use the official ROS2 Humble base image
FROM ros:humble

# Install necessary tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs \
    xvfb \
    python3-xvfbwrapper \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /ros2_ws

# Copy the package into the workspace
COPY . /ros2_ws/src/my_image_service

# Install Python dependencies from setup.py
RUN /bin/bash -c "cd /ros2_ws/src/my_image_service && python3 setup.py install"

# Source the ROS2 environment and build the package
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# Source the setup file
RUN /bin/bash -c "echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc"
RUN /bin/bash -c "echo 'source /ros2_ws/install/setup.bash' >> /root/.bashrc"

# Create the directory for saving images
RUN mkdir -p /images

# Run the node directly
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && /ros2_ws/install/my_image_service/bin/image_service"]
