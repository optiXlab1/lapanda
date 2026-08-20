#ifndef PANDA_EVAL_H
#define PANDA_EVAL_H

#include "../globals/globals.h"
#include "panda_types.h"

/*
 * Initialize and cleanup internal work buffers used by the evaluator.
 */
int panda_eval_init(unsigned int dimension_problem);
int panda_eval_cleanup(void);

/*
 * Estimate an initial gamma using a local numerical Lipschitz estimate:
 * gamma0 = alpha / L_est.
 */
real_t panda_eval_initial_gamma(const real_t* x, real_t alpha);

/*
 * Allocate/free memory for previous and current states.
 */
int panda_alloc_prev_state(panda_prev_state* state, unsigned int n);
int panda_free_prev_state(panda_prev_state* state);

int panda_alloc_current_state(panda_current_state* state, unsigned int n);
int panda_free_current_state(panda_current_state* state);

/*
 * Evaluate all quantities associated with a previous accepted point x and gamma.
 * This fills x, f_x, df_x, z, g_z, f_z, df_z, res, phi.
 */
int panda_eval_prev(
    const real_t* x,
    real_t gamma,
    real_t alpha,
    panda_prev_state* out
);

/*
 * Evaluate all quantities associated with a current trial point x and gamma.
 * This fills x, f_x, df_x, z, g_z, f_z, df_z, res, phi, upper_ok.
 */
int panda_eval_current(
    const real_t* x,
    real_t gamma,
    real_t alpha,
    panda_current_state* out
);

#endif