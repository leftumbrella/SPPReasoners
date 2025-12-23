cmake_minimum_required(VERSION 4.2)
set(TARGET_NAME spp)
project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
        onnxruntime::onnxruntime
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_imgcodecs
)
