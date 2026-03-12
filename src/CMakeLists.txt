
cmake_minimum_required(VERSION 3.10)
project(CVAlgorithms)

# 设置所有输出文件的路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/libsAlgorithms)
add_subdirectory(libsAlgorithms)
add_subdirectory(libsAlgorithmsTest)

