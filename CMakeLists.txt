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
)
target_link_libraries(geosolve
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

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

# For reference, commands when compiling cython module with distutils are:
# x86_64-linux-gnu-gcc -pthread -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC -I/usr/include/python3.4m -c model0.cpp -o build/temp.linux-x86_64-3.4/model0.o
#br x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 build/temp.linux-x86_64-3.4/model0.o -o /home/victor/Documents/DTU/S4-MasterThesis/code/src/model0.cpython-34m.so
