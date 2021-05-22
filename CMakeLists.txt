cmake_minimum_required(VERSION 3.19)
project(BP_iris)

set(CMAKE_CXX_STANDARD 11)

INCLUDE_DIRECTORIES(
        include
)
# 编译静态库
add_library(BPNet STATIC src/BPNet.cpp)

add_executable(BP_iris main.cpp)
# 链接静态库
target_link_libraries(BP_iris BPNet)