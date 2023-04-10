cmake_minimum_required(VERSION 3.22.0)

set(This ai_framework_bin)

project(${This} VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

#set(HeaderFiles)

set(SourceFiles
    main.cpp
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcublas -lcublas_device -lcudadevrt")

CUDA_ADD_EXECUTABLE(ai_framework_bin main.cpp )
target_link_libraries(ai_framework_bin ai_framework_lib)
set_target_properties(ai_framework_bin PROPERTIES OUTPUT_NAME ai_framework_bin)
set_target_properties(ai_framework_bin PROPERTIES CUDA_ARCHITECTURES "75")

set_target_properties(${This} PROPERTIES CUDA_ARCHITECTURES "75")
set_property(TARGET ${This} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${This} PROPERTY C_STANDARD 17)
set_property(TARGET ${This} PROPERTY CUDA_STANDARD 17)