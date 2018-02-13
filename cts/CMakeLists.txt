cmake_minimum_required(VERSION 3.2)
project(context_weighting_tree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gstabs -O3 -std=c++11 -pthread")

set(SOURCE_FILES
        cts_node.h cts_node.cpp
        cts.h cts.cpp
        location_dependent_cts.h location_dependent_cts.cpp
        python_api.cpp
        )

add_library(cwt SHARED ${SOURCE_FILES})