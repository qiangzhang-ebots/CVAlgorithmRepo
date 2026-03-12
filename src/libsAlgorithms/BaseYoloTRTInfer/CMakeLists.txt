
set(SUBPROJECT BaseYoloInfer)
file(GLOB SOURCES "*.cpp" "*.c", "*.hpp")
file(GLOB HEADERS "*.h")

# Find TensorRT
set(TENSORRT_ROOT "/home/ebots/Desktop/zhq/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8/TensorRT-10.8.0.43/targets/x86_64-linux-gnu") # Update this path to your TensorRT installation
# set(TENSORRT_ROOT "/root/perception/workspace/TRTProject/TensorRT-10.8.0.43.Linux.x86_64-gnu.cuda-12.8/TensorRT-10.8.0.43/targets/x86_64-linux-gnu") # Update this path to your TensorRT installation
set(TensorRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TensorRT_LIBRARY_DIRS "${TENSORRT_ROOT}/lib")
set(TensorRT_LIBRARIES "nvinfer;nvonnxparser;nvparsers;nvrtc;nvuffparser") # Update with the actual libraries you need

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# Add TensorRT library directory
# link_directories(${TensorRT_LIBRARY_DIRS})

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
