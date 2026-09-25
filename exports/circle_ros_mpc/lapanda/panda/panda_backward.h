#ifndef PANDA_BACKWARD_H
#define PANDA_BACKWARD_H

#include <stddef.h>

#include "../globals/globals.h"

typedef enum {
    PANDA_BACKWARD_SOLVER_AUTO = 0,
    PANDA_BACKWARD_SOLVER_MINRES = 1,
    PANDA_BACKWARD_SOLVER_GMRES = 2,
    PANDA_BACKWARD_SOLVER_CG = 3
} panda_backward_solver_type;

/*
 * C version of panda_backward.m.
 *
 * It computes:
 *   L, dL/dtheta
 * from u_star, theta, variable, and the final forward gamma.
 *
 * The callbacks are dispatched through function_evaluator.c from the unified
 * panda_oracle configured in optimizer_problem.
 */

typedef struct {
    unsigned char enable;
    real_t tol;
    unsigned int max_iter;
    unsigned int restart;
    real_t sym_tol;
    unsigned int sym_num_tests;
    panda_backward_solver_type force_solver;
    unsigned char recover_active;
    const real_t* initial_adjoint;
    real_t* output_adjoint;
} panda_backward_options;

int panda_backward_compute(
    const panda_backward_options* opts,
    const real_t* u_star,
    real_t gamma,
    real_t* dLdtheta,
    real_t* final_residual,
    unsigned int* iterations
);

/* Peak solver workspace held simultaneously by the most recent backward call. */
size_t panda_backward_get_last_peak_workspace_bytes(void);

/* Solver selected by the most recent backward call and whether CG fell back. */
panda_backward_solver_type panda_backward_get_last_solver_used(void);
unsigned char panda_backward_get_last_fallback_used(void);

#endif
