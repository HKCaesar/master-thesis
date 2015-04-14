cmake_minimum_required(VERSION 2.8)

project(MasterThesis)

find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(Ceres REQUIRED PATHS "ceres")

include_directories(${CERES_INCLUDE_DIRS})
include_directories("../cereal-1.1.1/include")

# Features analysis
add_executable(features_analysis
    src/features_analysis.cpp
    )
target_link_libraries(features_analysis ${OpenCV_LIBS})

# Model0
add_executable(model0
    src/model0.cpp
)

target_link_libraries(model0
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
