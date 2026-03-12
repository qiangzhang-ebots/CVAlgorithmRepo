
cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME CVAlgorithm)
project(${PROJECT_NAME})

# 设置所有输出文件的路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/libsAlgorithms)
add_subdirectory(libsAlgorithms)
add_subdirectory(libsAlgorithmsTest)

set(modules 
	BaseYoloTRTInfer
    FPCYoloTRTInfer
)

# --- install ---
install(TARGETS ${modules}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY libsAlgorithms/ DESTINATION include/
    FILES_MATCHING PATTERN "*.h"
)

# --- ament export ---
find_package(ament_cmake REQUIRED)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(${modules})

ament_package()

