cmake_minimum_required(VERSION 3.22.0)

set(This ai_framework)
set(ThisLib ${This}_lib)

project(${ThisLib} VERSION 0.1.0 LANGUAGES CUDA CXX)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(HeaderFiles
    src/matrix.cuh
)

set(SourceFiles
    src/matrix.cu
)

add_library(${ThisLib} STATIC ${SourceFiles} ${HeaderFiles})

CUDA_ADD_EXECUTABLE(${This} src/main.cpp )
target_link_libraries(${ThisLib} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(${This} ${ThisLib})
set_target_properties(${This} PROPERTIES OUTPUT_NAME ai_framework)

set_target_properties(${ThisLib} PROPERTIES CUDA_ARCHITECTURES "75")
set_property(TARGET ${ThisLib} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${ThisLib} PROPERTY C_STANDARD 17)

#include(CPack)
include(CTest)
enable_testing()

add_subdirectory(externals/googletest)
add_subdirectory(tests)
#add_subdirectory(src)