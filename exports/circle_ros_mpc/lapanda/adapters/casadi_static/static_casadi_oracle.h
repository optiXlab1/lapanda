#ifndef LAPANDA_STATIC_CASADI_ORACLE_H
#define LAPANDA_STATIC_CASADI_ORACLE_H

#include "../../alm/alm.h"
#include "../../include/optimizer.h"

void lapanda_static_init_panda_problem(
    struct optimizer_problem* problem,
    const struct solver_parameters* solver_params,
    const struct backward_parameters* backward_params
);

void lapanda_static_init_alm_problem(
    alm_problem* problem,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    const struct solver_parameters* inner_solver_params,
    const struct backward_parameters* backward_params
);

#endif
