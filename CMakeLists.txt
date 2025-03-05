cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW) # Set policy
project(threat_detector VERSION 1.0.0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-attributes -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-attributes -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# Find OpenCV
find_package(OpenCV 3.4 REQUIRED)   # sudo apt-get install libopencv-dev

# Find Boost
find_package(Boost REQUIRED COMPONENTS system) # sudo apt-get install libboost-all-dev

# Find Eigen
find_package(Eigen3 3.3.4 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PYTHON_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/pybind11/include
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

# You need to set to your own Python paths
add_subdirectory(${PROJECT_SOURCE_DIR}/pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
include/System.h
include/tcp_interface.hpp
src/System.cc
)

target_link_libraries(${PROJECT_NAME}
${CONDA_LIBS}
pybind11::embed
${OpenCV_LIBS}
${EIGEN3_LIBS}
-lboost_serialization
-lboost_system
-lcrypto
-lpthread
#-lncurses   # for Jetson
#-lncursesw   # for Jetson
-lffi   # for Jetson
snark_imaging   # Mission systems
comma_csv  # Mission systems
comma_string  # Mission systems
comma_xpath  # Mission systems
)

### Build main demos
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src)
add_executable(detect src/detect.cpp)
target_link_libraries(detect ${PROJECT_NAME})

