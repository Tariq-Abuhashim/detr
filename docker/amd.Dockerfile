FROM missionsystems/base:20.04-ros2-cuda-devel

### Create necessary directories and set workdir
RUN mkdir -p /home/devops/Thirdparty /home/devops/src/hyperteaming
WORKDIR /home/devops/Thirdparty

### Install OpenCV from apt repository
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv
USER devops

### Install PyTorch from source
WORKDIR /home/devops/src
RUN git clone --recursive -b v1.11.0 https://github.com/pytorch/pytorch
WORKDIR /home/devops/src/pytorch
RUN git submodule sync && git submodule update --init --recursive --jobs 0
USER root
RUN python3 -m pip install astunparse numpy==1.19.5 ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN CMAKE_CUDA_ARCHITECTURES="75" \
    python3 setup.py install

### Install TorchVision from source
USER devops
WORKDIR /home/devops/src
RUN git clone -b v0.12.0 https://github.com/pytorch/vision.git
WORKDIR /home/devops/src/vision
USER root
RUN python3 setup.py install

### Additional setup as the non-root user
USER devops
WORKDIR /home/devops/src/installers
RUN ./install.sh --install-role=eigen --vars ansible_sudo_pass=devops
RUN ./install.sh --install-role=comma --vars ansible_sudo_pass=devops
USER root
RUN rm -rf /usr/local/lib/libtbb.so
USER devops
RUN ./install.sh --install-role=snark --vars snark_cmake_opts="-Dsnark_build_ros_version_1=OFF -Dsnark_build_imaging=ON -Dsnark_build_ros_version_2=ON -Dsnark_build_imaging_opencv_contrib=OFF -Dsnark_build_navigation=ON -Dsnark_build_sensors_dc1394=OFF -Dsnark_build_sensors_ouster=ON -Dsnark_build_ros=ON -DCXX_STANDARD_TO_USE=17",ansible_sudo_pass=devops,ros_version=ros2
# RUN ./install.sh --install-role=snark --vars snark_cmake_opts="-Dsnark_build_ros_version_1=OFF -Dsnark_build_ros_version_2=ON -Dsnark_build_imaging_opencv_contrib=OFF -Dsnark_build_navigation=ON -Dsnark_build_sensors_dc1394=OFF -Dsnark_build_sensors_ouster=ON -Dsnark_build_ros=ON -DCXX_STANDARD_TO_USE=17",ansible_sudo_pass=devops,ros_version=ros2

### TensorRT and Pycuda
USER root
RUN python3 -m pip install tensorrt
RUN apt install -y tensorrt
RUN apt install -y python3-pycuda
RUN apt install -y python3-tk
RUN apt install -y git-lfs

### Ros2
RUN apt-get update && apt-get install -y \
    ros-humble-rclpy\
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs

### DETR
WORKDIR /home/devops/src/hyperteaming
USER devops
RUN git clone git@gitlab.com:missionsystems/hyperteaming/detr.git
RUN pip3 install cython scipy
RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip3 install git+https://github.com/cocodataset/panopticapi.git




USER root
RUN echo "deb [arch=amd64 trusted=yes] http://100.118.78.234/missionsystems_ppa stable main" | sudo tee /etc/apt/sources.list.d/mission_systems_ppa.list
RUN apt update
RUN apt install -y ros-humble-ms-civtak-ros-bridge-msgs

USER devops

#USER devops
#WORKDIR /home/devops/src/installers

#RUN ./install.sh --install-role=snark --vars ansible_sudo_pass=devops
#RUN ./install.sh --install-role=innovusion --vars ansible_sudo_pass=devops

#USER root

# # Clone DETR and install requirements
# WORKDIR /home/devops/src/hyperteaming
# USER devops
# RUN git clone git@gitlab.com:missionsystems/hyperteaming/detr.git
# WORKDIR /home/devops/src/hyperteaming/detr
# RUN git lfs install
# RUN git lfs pull
# RUN mkdir -p /home/devops/src/hyperteaming/detr/build 
# WORKDIR /home/devops/src/hyperteaming/detr/build
# RUN cmake .. && make -j 

# # Serialise the model and run the node
# WORKDIR /home/devops/src/hyperteaming/detr/src
#RUN python3 build_model.py
#RUN python3 serialise_engine.py
#source /opt/ros/humble/setup.bash 
#RUN python3 detect_service.py

#./detect ../configs/vulcan.yaml
# [CLASS] [PROB] [BOXx] [BOXy] [BOXw] [BOXh]

# Probably need to add /usr/src/tensorrt/bin to path so we have /usr/src/tensorrt/bin/trtexec
# Needs CUDA TO RUN.
# RUN /usr/src/tensorrt/bin/trtexec --onnx=detr.onnx --saveEngine=detr.trt

# python3 -m pip install tensorrt
# sudo apt install python3-pycuda
# sudo apt install python3-tk

# ./install.sh --install-role=snark --vars snark_cmake_opts="-Dsnark_build_ros_version_1=OFF -Dsnark_build_ros_version_2=ON -Dsnark_build_imaging_opencv_contrib=OFF -Dsnark_build_navigation=ON -Dsnark_build_sensors_dc1394=OFF -Dsnark_build_sensors_ouster=ON -Dsnark_build_ros=ON"
# sudo apt install python3-astunparse
# sudo apt install python3-ninja
# sudo apt install python3-pyyaml
# sudo apt install python3-mkl
# sudo apt install python3-mkl-include
# sudo apt install python3-setuptools
# sudo apt install python3-cmake
# sudo apt install python3-cffi
# sudo apt install python3-future
# sudo apt install python3-dataclasses

# These work with apt
# sudo apt install python3-typing-extensions
# sudo apt install python3-numpy
# sudo apt install python3-six
# sudo apt install python3-requests


# RUN ./install.sh --install-role=snark --vars snark_cmake_opts="
# -Dsnark_build_ros_version_1=OFF
# -Dsnark_build_ros_version_2=ON
# -Dsnark_build_imaging_opencv_contrib=OFF
# -Dsnark_build_navigation=ON
# -Dsnark_build_sensors_dc1394=OFF
# -Dsnark_build_sensors_ouster=ON
# -Dsnark_build_ros=ON
# -DCXX_STANDARD_TO_USE=17


# -DCMAKE_BUILD_TYPE=Release \
# -DCXX_STANDARD_TO_USE=17 \
# -DBUILD_SHARED_LIBS=ON \
# -DBUILD_TESTS=OFF \
# -DINSTALL_TESTS=OFF \
# -Dsnark_build_graphics=OFF \
# -Dsnark_build_graphics_csv_plot=OFF \
# -Dsnark_build_imaging=OFF \
# -Dsnark_build_imaging_opencv_contrib=OFF \
# -Dsnark_build_ros=OFF \
# -Dsnark_build_ros_version_1=OFF \
# -Dsnark_build_ros_version_2=OFF \
# -Dsnark_build_sensors_dc1394=OFF \
# -Dsnark_build_sensors_hokuyo=OFF \
# -Dsnark_build_sensors_sick=OFF


# ",ansible_sudo_pass=devops,ros_version=ros2

# sudo python3 -m pip install tensorrt
# sudo apt install -y python3-pycuda
# sudo apt install -y python3-tk
# sudo apt install -y git-lfs
