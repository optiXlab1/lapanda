set(PANDA_CORE_SRC
    function_evaluator.c
    matrix_operations.c
    lbfgs.c
    panda_eval.c
    panda_forward.c
    panda_linear_solver.c
    panda_backward.c
    optimizer.c
)

file(GLOB CONSTRAINT_SRC
    "constraints/*.c"
)

add_library(PANDA_lib STATIC
    ${PANDA_CORE_SRC}
    ${CONSTRAINT_SRC}
)

set_property(TARGET PANDA_lib PROPERTY C_STANDARD 90)
set_property(TARGET PANDA_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(PANDA_fake STATIC
    matrix_operations.c
    lbfgs.c
    panda_eval.c
    panda_forward.c
    panda_linear_solver.c
    panda_backward.c
)

set_property(TARGET PANDA_fake PROPERTY C_STANDARD 90)
