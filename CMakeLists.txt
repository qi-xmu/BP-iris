cmake_minimum_required(VERSION 3.19)
project(BP_iris)

set(CMAKE_CXX_STANDARD 11)

add_executable(BP_iris main.cpp lib/BPNet/BPNet.cpp lib/BPNet/BPNet.h lib/Dateset/Dateset.cpp lib/Dateset/Dateset.h lib/Node/Node.cpp lib/Node/Node.h)