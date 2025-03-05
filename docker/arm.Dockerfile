FROM missionsystems/base:20.04-ros2-cuda-devel

# Create necessary directories and set workdir
RUN mkdir -p /home/devops/Thirdparty /home/devops/src/hyperteaming
WORKDIR /home/devops/Thirdparty

# Clone and build OpenCV
RUN git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git
RUN mkdir -p /home/devops/Thirdparty/opencv/build
WORKDIR /home/devops/Thirdparty/opencv/build
RUN cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=OFF  \
  -DBUILD_DOCS=OFF  \
  -DBUILD_PACKAGE=OFF \
  -DBUILD_TESTS=OFF  \
  -DBUILD_PERF_TESTS=OFF  \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_calib3d=ON  \
  -DBUILD_opencv_cudaoptflow=OFF  \
  -DBUILD_opencv_dnn=OFF  \
  -DBUILD_opencv_dnn_BUILD_TORCH_IMPORTER=OFF  \
  -DBUILD_opencv_features2d=ON \
  -DBUILD_opencv_flann=ON \
  -DBUILD_opencv_java=ON  \
  -DBUILD_opencv_objdetect=ON  \
  -DBUILD_opencv_python2=OFF  \
  -DBUILD_opencv_python3=OFF  \
  -DBUILD_opencv_photo=ON \
  -DBUILD_opencv_stitching=ON  \
  -DBUILD_opencv_superres=ON  \
  -DBUILD_opencv_shape=ON  \
  -DBUILD_opencv_videostab=OFF \
  -DBUILD_PROTOBUF=OFF \
  -DWITH_1394=OFF  \
  -DWITH_GSTREAMER=OFF  \
  -DWITH_GPHOTO2=OFF  \
  -DWITH_MATLAB=OFF  \
  -DWITH_NVCUVID=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_OPENCLAMDBLAS=OFF \
  -DWITH_OPENCLAMDFFT=OFF \
  -DWITH_TIFF=OFF  \
  -DWITH_VTK=OFF  \
  -DWITH_WEBP=OFF  \
  ..

RUN make -j$(nproc)
USER root
RUN make install
USER devops

# Install PyTorch from source
WORKDIR /home/devops/src
RUN git clone --recursive -b v1.11.0 https://github.com/pytorch/pytorch
WORKDIR /home/devops/src/pytorch
RUN git submodule sync && git submodule update --init --recursive --jobs 0

USER root
#RUN python3 -m pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN python3 -m pip install astunparse numpy ninja pyyaml setuptools cmake typing_extensions future six requests dataclasses
RUN CMAKE_CUDA_ARCHITECTURES="75" \
    python3 setup.py install

# Install TorchVision from source
USER devops
WORKDIR /home/devops/src
RUN git clone -b v0.12.0 https://github.com/pytorch/vision.git
WORKDIR /home/devops/src/vision
USER root
RUN python3 setup.py install

# # Additional setup as the non-root user
USER devops
WORKDIR /home/devops/src/installers
RUN ./install.sh --install-role=eigen --vars ansible_sudo_pass=devops
RUN ./install.sh --install-role=comma --vars ansible_sudo_pass=devops
USER root
RUN rm -rf /usr/local/lib/libtbb.so
USER devops
RUN ./install.sh --install-role=snark --vars snark_cmake_opts="-Dsnark_build_ros_version_1=OFF -Dsnark_build_imaging=ON -Dsnark_build_ros_version_2=ON -Dsnark_build_imaging_opencv_contrib=OFF -Dsnark_build_navigation=ON -Dsnark_build_sensors_dc1394=OFF -Dsnark_build_sensors_ouster=ON -Dsnark_build_ros=ON -DCXX_STANDARD_TO_USE=17",ansible_sudo_pass=devops,ros_version=ros2
# RUN ./install.sh --install-role=snark --vars snark_cmake_opts="-Dsnark_build_ros_version_1=OFF -Dsnark_build_ros_version_2=ON -Dsnark_build_imaging_opencv_contrib=OFF -Dsnark_build_navigation=ON -Dsnark_build_sensors_dc1394=OFF -Dsnark_build_sensors_ouster=ON -Dsnark_build_ros=ON -DCXX_STANDARD_TO_USE=17",ansible_sudo_pass=devops,ros_version=ros2

#WORKDIR /home/devops/src/hyperteaming/detr

USER root
RUN cd /home/devops/src/hyperteaming
#RUN python3 -m pip install tensorrt
#RUN apt install -y tensorrt
#RUN apt install -y python3-pycuda
RUN apt install -y python3-tk
RUN apt install -y git-lfs

RUN apt-get update && apt-get install -y \
    ros-humble-rclpy\
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-vision-msgs

RUN python3 -m pip install tensorrt


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
