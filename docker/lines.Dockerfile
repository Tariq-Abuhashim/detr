from missionsystems/hyperteaming:detection

# RUN sudo apt update && sudo apt install -y python3-pip

USER root
RUN apt update && sudo apt install -y colmap python3-pandas python3-pyproj python3-pybind11
RUN python3 -m pip install pycolmap
RUN python3 -m pip install pytlsd
RUN python3 -m pip install gluestick

RUN apt update && sudo apt install -y git libhdf5-dev build-essential cmake 


RUN apt-get update && apt-get install -y git cmake build-essential \
    libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \
    libboost-regex-dev libboost-system-dev libboost-test-dev \
    libeigen3-dev libsuitesparse-dev libfreeimage-dev libgoogle-glog-dev libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libsqlite3-dev\
    libatlas-base-dev libsuitesparse-dev libceres-dev libmetis-dev libhdf5-dev libflann-dev

USER devops
WORKDIR /home/devops/src
# RUN git clone https://github.com/colmap/colmap.git
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /home/devops/src/colmap
RUN git checkout 3.8
RUN mkdir -p build
WORKDIR /home/devops/src/colmap/build
RUN cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..
RUN make -j$(nproc)  # Compile using all available cores
USER root
RUN make install

USER devops
WORKDIR /home/devops/src
RUN git clone --recursive https://github.com/vlarsson/PoseLib.git
WORKDIR /home/devops/src/PoseLib
RUN mkdir -p build
WORKDIR /home/devops/src/PoseLib/build
RUN cmake ..
RUN make -j$(nproc)
USER root
RUN make install

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Create a new Conda environment named "limap" with Python 3.9 and CUDA Toolkit 11.4
RUN /opt/conda/bin/conda create -n limap python=3.9 cudatoolkit=11.8 -y


# SHELL ["/opt/conda/bin/conda", "run", "-n", "limap", "/bin/bash", "-c"]
# RUN "python.__version__"
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/bin/activate limap" >> ~/.bashrc
CMD ["bash", "-c", "source /opt/conda/bin/activate limap "]

COPY window-tracker /home/devops/src/window-tracker
WORKDIR /home/devops/src/window-tracker/line-mapping
RUN python3 -m pip install -r requirements.txt


# RUN /opt/conda/bin/conda init
# RUN echo "/opt/conda/bin/conda init" >> ~/.bashrc
# RUN echo "/opt/conda/bin/conda activate limap" >> ~/.bashrc
# USER devops

# RUN /opt/conda/bin/conda init
# RUN /opt/conda/bin/conda activate limap
# RUN conda activate
# RUN source activate limap


# WORKDIR /home/devops/src

# # RUN git clone git@gitlab.com:missionsystems/hyperteaming/window-tracker.git
# COPY window-tracker /home/devops/src/window-tracker
# WORKDIR /home/devops/src/window-tracker/limap/third-party

# RUN git clone https://github.com/cvg/Hierarchical-Localization.git
# WORKDIR /home/devops/src/window-tracker/limap/third-party/Hierarchical-Localization
# RUN git submodule update --init --recursive && cd ../

# RUN git clone https://github.com/B1ueber2y/JLinkage
# RUN git clone https://github.com/B1ueber2y/libigl.git
# RUN git clone https://github.com/B1ueber2y/RansacLib.git
# WORKDIR /home/devops/src/window-tracker/limap/third-party/RansacLib 
# RUN git submodule update --init --recursive && cd ../

# RUN git clone https://github.com/B1ueber2y/HighFive.git
# WORKDIR /home/devops/src/window-tracker/limap/third-party/HighFive
# RUN git submodule update --init --recursive && cd ../
# WORKDIR /home/devops/src/window-tracker/limap/third-party
# RUN git clone https://github.com/iago-suarez/pytlbd.git
# WORKDIR /home/devops/src/window-tracker/limap/third-party/pytlbd
# RUN git submodule update --init --recursive && cd ../
# WORKDIR /home/devops/src/window-tracker/limap/third-party
# RUN git clone https://github.com/cherubicXN/hawp.git
# RUN git clone https://github.com/cvg/DeepLSD.git
# WORKDIR /home/devops/src/window-tracker/limap/third-party/DeepLSD
# RUN git submodule update --init --recursive && cd ../
# RUN git clone https://github.com/rpautrat/TP-LSD.git

# #cd TP-LSD && git submodule update --init --recursive
# #cd tp_lsd/modeling && rm -r DCNv2
# WORKDIR /home/devops/src/window-tracker/limap/third-party/TP-LSD/tp_lsd/modeling
# RUN git clone https://github.com/lucasjinreal/DCNv2_latest.git DCNv2

# RUN rm -rf /usr/lib/python3/dist-packages/PyYAML-5.3.1.egg-info
# RUN python3 -m pip install tqdm
# RUN python3 -m pip install attrdict
# RUN python3 -m pip install h5py
# RUN python3 -m pip install seaborn
# RUN python3 -m pip install brewer2mpl
# RUN python3 -m pip install shapely
# RUN python3 -m pip install jupyter
# RUN python3 -m pip install bresenham
# RUN python3 -m pip install pyvista
# RUN python3 -m pip install omegaconf
# RUN python3 -m pip install rtree
# RUN python3 -m pip install plyfile
# RUN python3 -m pip install pathlib
# RUN python3 -m pip install open3d
# RUN python3 -m pip install imagesize
# RUN python3 -m pip install einops
# RUN python3 -m pip install yacs
# RUN python3 -m pip install python-json-logger
# RUN python3 -m pip install pytlsd
# RUN python3 -m pip install gluestick
# RUN python3 -m pip install pycolmap

# WORKDIR /home/devops/src/window-tracker/limap
# RUN python3 -m pip install -r requirements.txt

# WORKDIR /home/devops/src/PoseLib
# RUN python3 setup.py install


# WORKDIR /home/devops/src/window-tracker/limap

# # # # # # # # # # # # # # Fez - fehhat

