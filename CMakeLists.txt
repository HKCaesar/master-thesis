cmake_minimum_required(VERSION 2.8)

project(MasterThesis)

find_package(OpenCV REQUIRED)

add_executable(features
    src/features.cpp
    )

target_link_libraries(features ${OpenCV_LIBS})

