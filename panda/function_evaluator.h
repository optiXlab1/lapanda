#ifndef FUNCTION_EVALUATOR_H
#define FUNCTION_EVALUATOR_H

#include "../globals/globals.h"

struct optimizer_problem;

int function_evaluator_init(const struct optimizer_problem* problem);
int function_evaluator_cleanup(void);
int function_evaluator_set_parameters(
    const real_t* theta,
    const real_t* variable
);

real_t function_evaluator_f_df(const real_t* input, real_t* output);
real_t function_evaluator_proxg(real_t* input, real_t gamma);

int function_evaluator_jprox(
    const real_t* u_star,
    real_t gamma,
    real_t* J
);

int function_evaluator_hvp(
    const real_t* v,
    const real_t* u_star,
    real_t* Hv
);

int function_evaluator_loss_grad(
    const real_t* u_star,
    real_t* L,
    real_t* grad_u_L
);

int function_evaluator_vjp(
    const real_t* v,
    const real_t* u_star,
    real_t* grad_theta
);

unsigned int function_evaluator_get_n(void);
unsigned int function_evaluator_get_ntheta(void);
unsigned int function_evaluator_get_nvar(void);

#endif
