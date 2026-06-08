set(SUBPROJECT YoloInferTest)

add_executable(${SUBPROJECT} yolo_task_test.cpp)

target_link_libraries(${SUBPROJECT} PRIVATE
    YoloTRTInfer
    ${OpenCV_LIBS}
)

target_include_directories(${SUBPROJECT} PRIVATE
    ${PROJECT_SOURCE_DIR}/libsAlgorithms/YoloTRTInfer
)
