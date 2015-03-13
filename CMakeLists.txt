cmake_minimum_required(VERSION 2.8)

project(MasterThesis)

find_package(OpenCV REQUIRED)

add_executable(features_analysis
    src/features_analysis.cpp
    )

target_link_libraries(features_analysis ${OpenCV_LIBS})

