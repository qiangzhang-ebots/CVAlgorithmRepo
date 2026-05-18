set(SUBPROJECT LedBoardTRTInfer)
file(GLOB SOURCES "*.cpp" "*.c" "*.hpp")
file(GLOB HEADERS "*.h")

include_directories(/usr/local/cuda-12.8/include)
include_directories(/usr/include/opencv4)
add_library(${SUBPROJECT} SHARED ${SOURCES} ${HEADERS})

target_include_directories(${SUBPROJECT}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/LedBoardTRTInfer>
)

target_link_libraries(${SUBPROJECT} PUBLIC
    BaseYoloTRTInfer
)

install(TARGETS ${SUBPROJECT}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} 
    DESTINATION include/LedBoardTRTInfer
)
