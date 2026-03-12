
include_directories(/usr/local/cuda-12.8/include)
include_directories(/usr/include/opencv4)

# Similarly scan test subdirectories
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
            add_subdirectory(${child})
        endif()
    endif()
endforeach()
