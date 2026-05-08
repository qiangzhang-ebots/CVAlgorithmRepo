set(SUBPROJECT HRNetInferTest)

add_executable(${SUBPROJECT} main.cpp)

target_link_libraries(${SUBPROJECT} PRIVATE
    HRNetInfer
)
