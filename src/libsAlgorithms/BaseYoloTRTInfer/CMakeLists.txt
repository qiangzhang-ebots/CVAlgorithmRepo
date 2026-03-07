
set(SUBPROJECT BaseYoloInfer)
file(GLOB SOURCES "*.cpp" "*.c" "*.hpp")
file(GLOB HEADERS "*.h")

# Find TensorRT
set(TENSORRT_ROOT "/usr/local/lib/TensorRT-8.4.3.1") 
set(TensorRT_INCLUDE_DIRS "/usr/local/include/TensorRT-8.4.3.1")
set(TensorRT_LIBRARY_DIRS "${TENSORRT_ROOT}")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_library(${SUBPROJECT} SHARED ${SOURCES} ${HEADERS})


target_link_libraries(${SUBPROJECT} PUBLIC
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    "${TensorRT_LIBRARY_DIRS}/libnvinfer.so"
    "${TensorRT_LIBRARY_DIRS}/libnvonnxparser.so"
)

target_include_directories(${SUBPROJECT} PUBLIC
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
