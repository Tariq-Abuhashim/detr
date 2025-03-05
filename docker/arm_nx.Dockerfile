FROM missionsystems/base:20.04-ros2-cuda-devel
USER root
USER apt update
USER root
RUN apt install -y build-essential git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev \
    python3-dev python3-numpy libtbb2 libtbb-dev \
    libdc1394-22-dev
RUN apt install -y python3-pip
RUN apt update

RUN apt install -y python3-numpy
RUN apt install -y python3-libnvinfer-dev
RUN apt install -y tensorrt

RUN echo 'export PATH=/usr/src/tensorrt/bin:$PATH' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/src/tensorrt/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
RUN pip install pycuda

USER devops
