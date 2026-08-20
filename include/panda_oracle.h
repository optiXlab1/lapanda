#ifndef PANDA_ORACLE_H
#define PANDA_ORACLE_H

#include "../globals/globals.h"

typedef real_t (*panda_prox_fun)(
    real_t* input,
    real_t gamma,
    const real_t* theta,
    const real_t* variable
);

typedef real_t (*panda_cost_grad_fun)(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    real_t* output_gradient
);

typedef int (*panda_jprox_fun)(
    const real_t* u_star,
    real_t gamma,
    const real_t* theta,
    const real_t* variable,
    real_t* J
);

typedef int (*panda_hvp_fun)(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* Hv
);

typedef int (*panda_loss_grad_fun)(
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* L,
    real_t* grad_u_L
);

typedef int (*panda_vjp_fun)(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* grad_theta
);

typedef struct panda_oracle {
    unsigned int n;
    unsigned int ntheta;
    unsigned int nvar;

    panda_prox_fun proxg;
    panda_cost_grad_fun cost_gradient;

    panda_jprox_fun jprox;
    panda_hvp_fun hvp;
    panda_loss_grad_fun loss_grad;
    panda_vjp_fun vjp;
} panda_oracle;

#endif
