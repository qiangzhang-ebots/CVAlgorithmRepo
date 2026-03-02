
set(SUBPROJECT FPCYoloInfer)
file(GLOB SOURCES "*.cpp" "*.c", "*.hpp")
file(GLOB HEADERS "*.h")

add_library(${SUBPROJECT} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${SUBPROJECT} PUBLIC
    BaseYoloInfer
)
