### make sure the user is doing an out-of-source build:
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed. use cmake -H. -Bbuild instead to build in the build folder")
endif()
###

cmake_minimum_required(VERSION 3.15)
project(lapanda C)

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    if(NOT MSVC)
        set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fprofile-arcs -ftest-coverage")
        set(LDFLAGS "-O0 -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
    endif()
endif()

# subdirectory with PANDA library
include_directories("${PROJECT_SOURCE_DIR}/panda")
include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(panda)
add_subdirectory(alm)

option(BUILD_PYTHON "Build the pybind11 Python bindings" OFF)
if(BUILD_PYTHON)
    enable_language(CXX)
    add_subdirectory(python)
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
