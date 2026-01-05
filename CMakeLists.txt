cmake_minimum_required(VERSION 4.2)
set(TARGET_NAME spp)
project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(${TARGET_NAME} main.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
        CUDA::cudart
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_imgcodecs
)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR
        NvInfer.h
        PATHS /usr/include
)

find_library(TENSORRT_LIB
        nvinfer
        PATHS /usr/lib/x86_64-linux-gnu
)

find_library(TENSORRT_PLUGIN_LIB
        nvinfer_plugin
        PATHS /usr/lib/x86_64-linux-gnu
)

target_include_directories(${TARGET_NAME} PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(${TARGET_NAME} PRIVATE
        ${TENSORRT_LIB}
        ${TENSORRT_PLUGIN_LIB}
)