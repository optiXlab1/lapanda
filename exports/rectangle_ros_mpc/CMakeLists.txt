cmake_minimum_required(VERSION 3.15)
project(lapanda_embedded_example C)

set(LAPANDA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/lapanda)

add_library(lapanda_embedded STATIC
    generated/rectangle_mpc_oracle.c
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

if(UNIX)
    target_link_libraries(lapanda_embedded PUBLIC m)
endif()

target_include_directories(lapanda_embedded PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
    ${LAPANDA_ROOT}/adapters/casadi_static
    ${LAPANDA_ROOT}/include
    ${LAPANDA_ROOT}/panda
    ${LAPANDA_ROOT}/alm
    ${LAPANDA_ROOT}/globals
)

add_executable(lapanda_embedded_demo main_example.c)
target_link_libraries(lapanda_embedded_demo PRIVATE lapanda_embedded)

add_executable(rectangle_mpc_benchmark main_rectangle_benchmark.c)
target_link_libraries(rectangle_mpc_benchmark PRIVATE lapanda_embedded)

add_executable(rectangle_imitation_benchmark main_rectangle_imitation_benchmark.c)
target_link_libraries(rectangle_imitation_benchmark PRIVATE lapanda_embedded)
