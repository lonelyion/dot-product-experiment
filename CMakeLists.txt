cmake_minimum_required(VERSION 3.17)
project(dot_product LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(dot_product main.cu my_dot_product.h)

set_target_properties(
        dot_product
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)