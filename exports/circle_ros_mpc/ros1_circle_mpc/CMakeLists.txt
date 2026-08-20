cmake_minimum_required(VERSION 3.10)
project(ros1_circle_mpc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

catkin_package()

set(LAPANDA_EXPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LAPANDA_ROOT ${LAPANDA_EXPORT_DIR}/lapanda)

add_library(lapanda_circle STATIC
    ${LAPANDA_EXPORT_DIR}/generated/circle_mpc_oracle.c
    ${LAPANDA_ROOT}/adapters/casadi_static/static_casadi_oracle.c
    ${LAPANDA_ROOT}/panda/function_evaluator.c
    ${LAPANDA_ROOT}/panda/matrix_operations.c
    ${LAPANDA_ROOT}/panda/lbfgs.c
    ${LAPANDA_ROOT}/panda/panda_eval.c
    ${LAPANDA_ROOT}/panda/panda_forward.c
    ${LAPANDA_ROOT}/panda/panda_linear_solver.c
    ${LAPANDA_ROOT}/panda/panda_backward.c
    ${LAPANDA_ROOT}/panda/optimizer.c
    ${LAPANDA_ROOT}/alm/alm.c
)

target_include_directories(lapanda_circle PUBLIC
    ${LAPANDA_EXPORT_DIR}/generated
    ${LAPANDA_ROOT}/adapters/casadi_static
    ${LAPANDA_ROOT}/include
    ${LAPANDA_ROOT}/panda
    ${LAPANDA_ROOT}/alm
    ${LAPANDA_ROOT}/globals
)

if(UNIX)
    target_link_libraries(lapanda_circle PUBLIC m)
endif()

add_executable(circle_mpc_node src/circle_mpc_node.cpp)
target_include_directories(circle_mpc_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(circle_mpc_node
    lapanda_circle
    ${catkin_LIBRARIES}
)
add_dependencies(circle_mpc_node ${catkin_EXPORTED_TARGETS})
