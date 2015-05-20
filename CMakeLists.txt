cmake_minimum_required(VERSION 2.8)

project(MasterThesis)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wdouble-promotion -Wfatal-errors")

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

# GeoSolve
add_executable(geosolve
    src/geosolve.cpp
    src/image_features.cpp
    src/model0.cpp
    src/model_terrain.cpp
    src/bootstrap.cpp
)
target_link_libraries(geosolve
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# Unit tests
include_directories("../gtest-1.7.0/include")
find_library(GTESTLIB gtest "../gtest-1.7.0/build")
add_executable(unittests
    unittests/types.cpp
)
target_link_libraries(unittests ${GTESTLIB} ${CMAKE_THREAD_LIBS_INIT})

# Cython module
# Should use find_package(PythonLibs 3), but couldn't get it to work
# so setting manually for now
include_directories("/usr/include/python3.4m")

add_library(pymodel0 SHARED
    cython_model0.cpp
)

target_link_libraries(pymodel0
    ${CERES_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# Run cython compiler before compiling the module
# Note, the DEPENDS argument here will only work with makefiles
# so other build systems will need to run cython manually
add_custom_command(OUTPUT cython_model0.cpp
    COMMAND cython --cplus ../cython/pymodel0.pyx -o cython_model0.cpp
    DEPENDS "${CMAKE_SOURCE_DIR}/cython/pymodel0.pyx"
    COMMENT "Compiling pymodel0.pyx"
)

set_target_properties(pymodel0 PROPERTIES COMPILE_FLAGS
    "-fwrapv -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -Wno-unused-function"
)

# Rename library file to follow python convention
add_custom_command(TARGET pymodel0 POST_BUILD
    COMMAND mv libpymodel0.so pymodel0.so
)
