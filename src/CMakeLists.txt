cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
    ScaleGANN
    LANGUAGES CXX CUDA)

# set(CMAKE_CUDA_STANDARD 17)

# find_package(CUDA REQUIRED)

# set(SOURCES main.cpp helper.cpp kernel.cu)

# add_executable(my_program ${SOURCES})

# set_target_properties(my_program PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "60"  # 设置 GPU 架构，例如 60 表示 sm_60
# )

# target_link_libraries(my_program ${CUDA_LIBRARIES})