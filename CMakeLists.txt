cmake_minimum_required(VERSION 4.1)
set(TARGET_NAME CppAi)
project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD 20)

add_executable(${TARGET_NAME} main.cpp)

target_include_directories(${TARGET_NAME} PRIVATE $ENV{OPENCV_DIR}include)
target_link_libraries(${TARGET_NAME} PUBLIC $ENV{OPENCV_DIR}x64/vc16/lib/opencv_world4120d.lib)

target_include_directories(${TARGET_NAME} PRIVATE $ENV{ONNXRUNTIME_DIR}CPU/include)
target_link_libraries(${TARGET_NAME} PUBLIC $ENV{ONNXRUNTIME_DIR}CPU/lib/onnxruntime.lib)
