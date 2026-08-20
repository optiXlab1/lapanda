#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../globals/globals.h"
#include "panda_oracle.h"
#include "../panda/panda_backward.h"

/* solver parameters */
struct solver_parameters {
    unsigned int max_iterations;   /* maximum PANOC+ iterations */
    real_t       tolerance;        /* stopping tolerance */
    unsigned int buffer_size;      /* L-BFGS memory size */
    unsigned int max_stable_iter;  /* used by gamma enlargement */
    unsigned char verbose;         /* write solver traces when nonzero */
};

struct backward_parameters {
    unsigned char enable;          /* whether backward calls are allowed */
    real_t tolerance;              /* linear solver tolerance */
    unsigned int max_iterations;   /* maximum linear solver iterations */
};

typedef struct {
    unsigned int iterations;
    real_t final_residual;
} optimizer_solve_info;

typedef void (*optimizer_trace_fun)(
    void* context,
    unsigned int iteration,
    real_t residual
);

/* optimization problem */
struct optimizer_problem {
    panda_oracle oracle;
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    optimizer_trace_fun trace;
    void* trace_context;
};

/* initialize optimizer */
int optimizer_init(struct optimizer_problem* problem_);

/* initialize optimizer with user-provided proximal operator */
int optimizer_init_with_custom_constraint(
    struct optimizer_problem* problem_,
    panda_prox_fun proxg
);

/* cleanup optimizer */
int optimizer_cleanup(void);

/* solve problem for the current theta/variable, overwriting solution in-place */
int solve_problem(
    real_t* solution,
    const real_t* theta,
    const real_t* variable
);

/* compute backward gradient through the accepted forward solution */
int solve_backward(
    const real_t* u_star,
    real_t* dLdtheta
);

int solve_backward_with_options(
    const real_t* u_star,
    const panda_backward_options* opts,
    real_t* dLdtheta
);

/* get final accepted gamma */
real_t get_gamma(void);

int optimizer_get_forward_info(optimizer_solve_info* info);
int optimizer_get_backward_info(optimizer_solve_info* info);

#endif /* OPTIMIZER_H */
