cmake_minimum_required(VERSION 3.0.2)
project(circle_acados_exact_ros1)

add_compile_options(-O2)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs)

set(ACADOS_SOURCE_DIR "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "acados source/install root")
if(NOT ACADOS_SOURCE_DIR)
    message(FATAL_ERROR "Set ACADOS_SOURCE_DIR to your acados install/source root")
endif()

catkin_package()

set(FORWARD_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")
set(SENS_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_sensitivity")
file(GLOB_RECURSE GENERATED_SOURCES CONFIGURE_DEPENDS
    "${FORWARD_GENERATED_DIR}/*.c"
    "${SENS_GENERATED_DIR}/*.c"
)
foreach(src ${GENERATED_SOURCES})
    get_filename_component(name "${src}" NAME)
    if(name MATCHES "^main_.*\\.c$" OR name MATCHES ".*sfunction.*\\.c$" OR name MATCHES ".*mex.*\\.c$")
        list(REMOVE_ITEM GENERATED_SOURCES "${src}")
    endif()
endforeach()

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${FORWARD_GENERATED_DIR}
    ${SENS_GENERATED_DIR}
    ${ACADOS_SOURCE_DIR}/include
    ${ACADOS_SOURCE_DIR}/include/acados
    ${ACADOS_SOURCE_DIR}/include/blasfeo/include
    ${ACADOS_SOURCE_DIR}/include/hpipm/include
)

link_directories(${ACADOS_SOURCE_DIR}/lib)

add_executable(circle_acados_node
    src/circle_acados_node.cpp
    ${GENERATED_SOURCES}
)

target_link_libraries(circle_acados_node
    ${catkin_LIBRARIES}
    acados hpipm blasfeo qpOASES_e m
)

set_target_properties(circle_acados_node PROPERTIES
    BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib"
    INSTALL_RPATH "${ACADOS_SOURCE_DIR}/lib"
)


