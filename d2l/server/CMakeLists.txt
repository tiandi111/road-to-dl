cmake_minimum_required(VERSION 3.16)
project(server)

set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(.
        /usr/local/include)

add_executable(server
    engine.cpp
    engine.h
    graph.cpp
    graph.h
    input_context.cpp
    input_context.h
    loader.cpp
    loader.h
    mkl.cpp
    mkl.h
    node.cpp
    node.h
    onnx.pb.cc
    onnx.pb.h
    tensor.cpp
    tensor.h
    utils.h)

target_link_directories(server PUBLIC /usr/local/lib)
target_link_libraries(server -lprotobuf)
target_link_libraries(server -ldnnl)