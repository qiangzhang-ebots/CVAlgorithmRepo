set(SUBPROJECT FPCYoloInferTest)

add_executable(${SUBPROJECT} main.cpp)

target_link_libraries(${SUBPROJECT} PRIVATE
    FPCYoloInfer
    BaseYoloInfer
)
