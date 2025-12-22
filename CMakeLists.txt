cmake_minimum_required(VERSION 4.2)
set(TARGET_NAME CppAi)
project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
        opencv_core
        onnxruntime::onnxruntime
)
