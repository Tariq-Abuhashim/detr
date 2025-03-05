#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a Python package is installed and get its version
python_package_version() {
    python3 -c "import $1; print($1.__version__)" 2>/dev/null
}

echo "Checking CUDA, TensorRT, PyCUDA, OpenCV, PyTorch, and torchvision installation..."

# Check if CUDA is installed
if command_exists nvcc; then
    echo "CUDA is installed."
    nvcc --version
else
    echo "CUDA is not installed."
fi

# Check if TensorRT is installed
if command_exists dpkg-query; then
    if dpkg-query -W -f='${Status}' tensorrt 2>/dev/null | grep -q "ok installed"; then
        echo "TensorRT is installed."
        tensorrt_version=$(dpkg-query -W -f='${Version}' tensorrt)
        echo "TensorRT version: $tensorrt_version"
    else
        echo "TensorRT is not installed."
    fi
else
    echo "dpkg-query is not installed. Cannot check TensorRT."
fi

# Check if PyCUDA is installed in Python
if command_exists python3; then
    pycuda_version=$(python_package_version pycuda)
    if [ -n "$pycuda_version" ]; then
        echo "PyCUDA is installed in Python."
        echo "PyCUDA version: $pycuda_version"
    else
        echo "PyCUDA is not installed in Python."
    fi
else
    echo "Python3 is not installed."
fi

# Check if OpenCV is installed as a system package using pkg-config
if command_exists pkg-config; then
    if pkg-config --exists opencv4; then
        echo "OpenCV system package is installed."
        opencv_version=$(pkg-config --modversion opencv4)
        echo "OpenCV version: $opencv_version"
    else
        echo "OpenCV system package is not installed."
    fi
else
    echo "pkg-config is not installed. Cannot check system packages."
fi

# Check if OpenCV is installed in Python
if command_exists python3; then
    opencv_python_version=$(python_package_version cv2)
    if [ -n "$opencv_python_version" ]; then
        echo "OpenCV is installed in Python."
        echo "OpenCV version in Python: $opencv_python_version"
    else
        echo "OpenCV is not installed in Python."
    fi
else
    echo "Python3 is not installed."
fi

# Check if PyTorch is installed in Python
if command_exists python3; then
    pytorch_version=$(python_package_version torch)
    if [ -n "$pytorch_version" ]; then
        echo "PyTorch is installed in Python."
        echo "PyTorch version: $pytorch_version"
    else
        echo "PyTorch is not installed in Python."
    fi
else
    echo "Python3 is not installed."
fi

# Check if torchvision is installed in Python
if command_exists python3; then
    torchvision_version=$(python_package_version torchvision)
    if [ -n "$torchvision_version" ]; then
        echo "torchvision is installed in Python."
        echo "torchvision version: $torchvision_version"
    else
        echo "torchvision is not installed in Python."
    fi
else
    echo "Python3 is not installed."
fi

