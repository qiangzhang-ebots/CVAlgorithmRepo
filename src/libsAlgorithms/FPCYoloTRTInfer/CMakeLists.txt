
set(SUBPROJECT FPCYoloTRTInfer)
file(GLOB SOURCES "*.cpp" "*.c" "*.hpp")
file(GLOB HEADERS "*.h")

include_directories(/usr/local/cuda-12.8/include)
include_directories(/usr/include/opencv4)
add_library(${SUBPROJECT} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${SUBPROJECT} PUBLIC
    BaseYoloTRTInfer
)
