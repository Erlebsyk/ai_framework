cmake_minimum_required(VERSION 3.22.0)

set(This ai_framework_tests)

set(SourceFiles
    matrix_test.cpp
	matrix_test.cu
    neural_net_test.cpp
    neural_net_test.cu
)

add_executable(${This} ${SourceFiles})

target_link_libraries(${This} PUBLIC
    gtest_main
    ai_framework_lib
)

add_test(
    NAME ${This}
    COMMAND ${This}
)