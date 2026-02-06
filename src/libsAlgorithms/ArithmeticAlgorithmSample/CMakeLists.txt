cmake_minimum_required(VERSION 3.8)
set(project_name "ArithmeticAlgorithmSample")
project(${project_name})

set(${project_name}_MAJOR_VERSION 0)
set(${project_name}_MINOR_VERSION 1)
set(${project_name}_PATCH_VERSION 0)
set(${project_name}_VERSION
  ${${project_name}_MAJOR_VERSION}.${${project_name}_MINOR_VERSION}.${${project_name}_PATCH_VERSION})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(${PROJECT_NAME} src/ArithmeticAlgorithm.cpp)

add_executable(${PROJECT_NAME}_Test test/main.cpp)
target_link_libraries(${PROJECT_NAME}_Test PRIVATE ${PROJECT_NAME})

# 关键 1: 必须在 target 上设置包含路径，并标记为 INTERFACE 或 PUBLIC
target_include_directories(${project_name} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


include(CMakePackageConfigHelpers)

# ✅ 使用默认模板里的 alg_add_VERSION
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake
  VERSION ${${project_name}_VERSION}
  COMPATIBILITY AnyNewerVersion
)


# 1. 安装库文件
install(TARGETS ${project_name}
  EXPORT ${project_name}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 2. 安装头文件
install(DIRECTORY include/
  DESTINATION include
)

# 3. 导出 Target 信息 (生成 alg_addTargets.cmake)
install(EXPORT ${project_name}Targets
  FILE ${project_name}Targets.cmake
  NAMESPACE ${project_name}::
  DESTINATION lib/cmake/${project_name}
)

# 4. 安装配置文件 (Config 和 ConfigVersion)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/${project_name}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${project_name}Config.cmake
  INSTALL_DESTINATION lib/cmake/${project_name}
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${project_name}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake
  DESTINATION lib/cmake/${project_name}
)