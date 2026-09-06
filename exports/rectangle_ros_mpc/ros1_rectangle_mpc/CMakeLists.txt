cmake_minimum_required(VERSION 3.10)
project(ros1_rectangle_mpc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

catkin_package()

set(lapanda_EXPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(lapanda_ROOT ${lapanda_EXPORT_DIR}/lapanda)

add_library(lapanda_rectangle STATIC
    ${lapanda_EXPORT_DIR}/generated/rectangle_mpc_oracle.c
    ${lapanda_ROOT}/adapters/casadi_static/static_casadi_oracle.c
    ${lapanda_ROOT}/panda/function_evaluator.c
    ${lapanda_ROOT}/panda/matrix_operations.c
    ${lapanda_ROOT}/panda/lbfgs.c
    ${lapanda_ROOT}/panda/panda_eval.c
    ${lapanda_ROOT}/panda/panda_forward.c
    ${lapanda_ROOT}/panda/panda_linear_solver.c
    ${lapanda_ROOT}/panda/panda_backward.c
    ${lapanda_ROOT}/panda/optimizer.c
    ${lapanda_ROOT}/alm/alm.c
)

target_include_directories(lapanda_rectangle PUBLIC
    ${lapanda_EXPORT_DIR}/generated
    ${lapanda_ROOT}/adapters/casadi_static
    ${lapanda_ROOT}/include
    ${lapanda_ROOT}/panda
    ${lapanda_ROOT}/alm
    ${lapanda_ROOT}/globals
)

if(UNIX)
    target_link_libraries(lapanda_rectangle PUBLIC m)
endif()

add_executable(rectangle_mpc_node src/rectangle_mpc_node.cpp)
target_include_directories(rectangle_mpc_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(rectangle_mpc_node
    lapanda_rectangle
    ${catkin_LIBRARIES}
)
add_dependencies(rectangle_mpc_node ${catkin_EXPORTED_TARGETS})
