#ifndef ALM_H
#define ALM_H

#include "../globals/globals.h"
#include "../include/optimizer.h"

typedef int (*alm_constraint_fun)(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    real_t* constraint
);

typedef int (*alm_constraint_jtprod_fun)(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multiplier,
    real_t* output_gradient
);

typedef int (*alm_constraint_hvp_fun)(
    const real_t* v,
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multipliers,
    const real_t* penalties,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    real_t* Hv
);

typedef int (*alm_constraint_vjp_fun)(
    const real_t* v,
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multipliers,
    const real_t* penalties,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    real_t* grad_theta
);

typedef struct {
    panda_oracle oracle;
    unsigned int ncon;
    const real_t* constraint_lower;
    const real_t* constraint_upper;

    alm_constraint_fun constraint;
    alm_constraint_jtprod_fun constraint_jtprod;
    alm_constraint_hvp_fun constraint_hvp;
    alm_constraint_vjp_fun constraint_vjp;

    struct solver_parameters inner_solver_params;
    struct backward_parameters backward_params;
} alm_problem;

typedef struct {
    unsigned int max_iterations;
    real_t tolerance;
    real_t initial_penalty;
    real_t penalty_update_factor;
    real_t max_penalty;
    real_t sufficient_decrease_factor;
    unsigned char verbose;
    unsigned char warm_start_inner;
} alm_parameters;

typedef struct {
    unsigned int iterations;
    real_t final_residual;
    real_t penalty;
    real_t forward_time_sec;
    real_t backward_time_sec;
} alm_info;

int alm_solve(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    const real_t* theta,
    const real_t* variable,
    alm_info* info
);

int alm_solve_with_backward(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info
);

int alm_solve_with_penalty0(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info
);

int alm_solve_with_backward_and_penalty0(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info
);

int alm_solve_with_backward_warm_start(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info,
    const real_t* adjoint0,
    real_t* adjoint_out
);

unsigned int alm_get_inner_iterations_count(void);
unsigned int alm_get_inner_iterations(unsigned int index);
unsigned int alm_get_penalties_count(void);
real_t alm_get_penalty(unsigned int index);

#endif
