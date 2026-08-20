#include "../globals/globals.h"
#include "../alm/alm.h"

#include <math.h>
#include <stdio.h>

static real_t identity_prox(
    real_t* input,
    real_t gamma,
    const real_t* theta,
    const real_t* variable
)
{
    (void)input;
    (void)gamma;
    (void)theta;
    (void)variable;
    return 0.0;
}

static real_t cost_gradient(
    const real_t* u,
    const real_t* theta,
    const real_t* variable,
    real_t* grad
)
{
    real_t d0;
    real_t d1;

    (void)variable;

    d0 = u[0] - theta[0];
    d1 = u[1] - theta[1];
    grad[0] = d0;
    grad[1] = d1;

    return 0.5 * (d0 * d0 + d1 * d1);
}

static int constraint_fun(
    const real_t* u,
    const real_t* theta,
    const real_t* variable,
    real_t* constraint
)
{
    (void)theta;
    (void)variable;

    constraint[0] = u[0] + u[1];
    return SUCCESS;
}

static int constraint_jtprod(
    const real_t* u,
    const real_t* theta,
    const real_t* variable,
    const real_t* multiplier,
    real_t* output_gradient
)
{
    (void)u;
    (void)theta;
    (void)variable;

    output_gradient[0] = multiplier[0];
    output_gradient[1] = multiplier[0];
    return SUCCESS;
}

static int free_jprox(
    const real_t* u_star,
    real_t gamma,
    const real_t* theta,
    const real_t* variable,
    real_t* J
)
{
    (void)u_star;
    (void)gamma;
    (void)theta;
    (void)variable;

    J[0] = 1.0;
    J[1] = 1.0;
    return SUCCESS;
}

static int identity_hvp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* Hv
)
{
    (void)u_star;
    (void)theta;
    (void)variable;

    Hv[0] = v[0];
    Hv[1] = v[1];
    return SUCCESS;
}

static int dummy_loss_grad(
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* L,
    real_t* grad_u_L
)
{
    (void)u_star;
    (void)theta;
    (void)variable;

    *L = 0.0;
    grad_u_L[0] = 0.0;
    grad_u_L[1] = 0.0;
    return SUCCESS;
}

static int dummy_vjp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* grad_theta
)
{
    (void)v;
    (void)u_star;
    (void)theta;
    (void)variable;

    grad_theta[0] = 0.0;
    grad_theta[1] = 0.0;
    return SUCCESS;
}

static real_t abs_max2(real_t a, real_t b)
{
    a = fabs(a);
    b = fabs(b);
    return (a > b) ? a : b;
}

int main(void)
{
    alm_problem problem;
    alm_parameters params;
    alm_info info;
    real_t theta[2];
    real_t variable[1];
    real_t solution[2];
    real_t lambda[1];
    real_t expected[2];
    real_t constraint_lower[1];
    real_t constraint_upper[1];
    real_t shift;
    int alm_iters;

    theta[0] = 2.0;
    theta[1] = -0.25;
    variable[0] = 0.0;
    solution[0] = 0.0;
    solution[1] = 0.0;
    lambda[0] = 0.0;
    constraint_lower[0] = 1.0;
    constraint_upper[0] = 1.0;

    shift = (theta[0] + theta[1] - 1.0) / 2.0;
    expected[0] = theta[0] - shift;
    expected[1] = theta[1] - shift;

    problem.oracle.n = 2;
    problem.oracle.ntheta = 2;
    problem.oracle.nvar = 1;
    problem.oracle.proxg = identity_prox;
    problem.oracle.cost_gradient = cost_gradient;
    problem.oracle.jprox = free_jprox;
    problem.oracle.hvp = identity_hvp;
    problem.oracle.loss_grad = dummy_loss_grad;
    problem.oracle.vjp = dummy_vjp;
    problem.ncon = 1;
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.constraint = constraint_fun;
    problem.constraint_jtprod = constraint_jtprod;

    problem.inner_solver_params.max_iterations = 100;
    problem.inner_solver_params.tolerance = 1e-8;
    problem.inner_solver_params.buffer_size = 5;
    problem.inner_solver_params.max_stable_iter = 0;
    problem.inner_solver_params.verbose = FALSE;

    params.max_iterations = 20;
    params.tolerance = 1e-6;
    params.initial_penalty = 10.0;
    params.penalty_update_factor = 5.0;
    params.max_penalty = 0.0;
    params.sufficient_decrease_factor = 0.25;
    params.verbose = FALSE;

    alm_iters = alm_solve(
        &problem,
        &params,
        solution,
        lambda,
        theta,
        variable,
        &info);
    if (alm_iters == FAILURE) {
        printf("alm_solve failed\n");
        return 1;
    }

    if (abs_max2(solution[0] - expected[0], solution[1] - expected[1]) > 1e-4 ||
        fabs(solution[0] + solution[1] - 1.0) > 1e-6 ||
        info.final_residual > params.tolerance) {
        printf("ALM equality QP failed\n");
        printf("solution=[%.16e %.16e] expected=[%.16e %.16e]\n",
               (double)solution[0], (double)solution[1],
               (double)expected[0], (double)expected[1]);
        printf("constraint=%.16e lambda=%.16e alm_res=%.16e iters=%u\n",
               (double)(solution[0] + solution[1] - 1.0),
               (double)lambda[0],
               (double)info.final_residual,
               info.iterations);
        return 1;
    }

    printf("ALM equality QP test passed\n");
    return 0;
}
