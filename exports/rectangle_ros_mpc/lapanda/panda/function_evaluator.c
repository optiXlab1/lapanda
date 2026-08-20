#include "../globals/globals.h"
#include "function_evaluator.h"
#include "../include/optimizer.h"

#include <stddef.h>

static const panda_oracle* g_oracle = NULL;
static const real_t* g_theta = NULL;
static const real_t* g_variable = NULL;

int function_evaluator_init(const struct optimizer_problem* problem)
{
    if (problem == NULL) {
        return FAILURE;
    }

    g_oracle = &problem->oracle;

    if (g_oracle->proxg == NULL || g_oracle->cost_gradient == NULL) {
        g_oracle = NULL;
        return FAILURE;
    }

    return SUCCESS;
}

int function_evaluator_cleanup(void)
{
    g_oracle = NULL;
    g_theta = NULL;
    g_variable = NULL;
    return SUCCESS;
}

int function_evaluator_set_parameters(
    const real_t* theta,
    const real_t* variable
)
{
    g_theta = theta;
    g_variable = variable;
    return SUCCESS;
}

real_t function_evaluator_f_df(const real_t* input, real_t* output)
{
    if (g_oracle == NULL || g_oracle->cost_gradient == NULL) {
        return 0.0;
    }

    return g_oracle->cost_gradient(input, g_theta, g_variable, output);
}

real_t function_evaluator_proxg(real_t* input, real_t gamma)
{
    if (g_oracle == NULL || g_oracle->proxg == NULL) {
        return 0.0;
    }

    return g_oracle->proxg(input, gamma, g_theta, g_variable);
}

int function_evaluator_jprox(
    const real_t* u_star,
    real_t gamma,
    real_t* J
)
{
    if (g_oracle == NULL || g_oracle->jprox == NULL) {
        return FAILURE;
    }

    return g_oracle->jprox(
        u_star,
        gamma,
        g_theta,
        g_variable,
        J);
}

int function_evaluator_hvp(
    const real_t* v,
    const real_t* u_star,
    real_t* Hv
)
{
    if (g_oracle == NULL || g_oracle->hvp == NULL) {
        return FAILURE;
    }

    return g_oracle->hvp(
        v,
        u_star,
        g_theta,
        g_variable,
        Hv);
}

int function_evaluator_loss_grad(
    const real_t* u_star,
    real_t* L,
    real_t* grad_u_L
)
{
    if (g_oracle == NULL || g_oracle->loss_grad == NULL) {
        return FAILURE;
    }

    return g_oracle->loss_grad(
        u_star,
        g_theta,
        g_variable,
        L,
        grad_u_L);
}

int function_evaluator_vjp(
    const real_t* v,
    const real_t* u_star,
    real_t* grad_theta
)
{
    if (g_oracle == NULL || g_oracle->vjp == NULL) {
        return FAILURE;
    }

    return g_oracle->vjp(
        v,
        u_star,
        g_theta,
        g_variable,
        grad_theta);
}

unsigned int function_evaluator_get_n(void)
{
    if (g_oracle == NULL) {
        return 0;
    }

    return g_oracle->n;
}

unsigned int function_evaluator_get_ntheta(void)
{
    if (g_oracle == NULL) {
        return 0;
    }

    return g_oracle->ntheta;
}

unsigned int function_evaluator_get_nvar(void)
{
    if (g_oracle == NULL) {
        return 0;
    }

    return g_oracle->nvar;
}
