#include "../alm/alm.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static real_t observed_penalty;

static real_t prox(real_t* x, real_t gamma, const real_t* theta, const real_t* var)
{
    (void)x; (void)gamma; (void)theta; (void)var;
    return 0.0;
}

static real_t cost(const real_t* x, const real_t* theta, const real_t* var, real_t* g)
{
    (void)var;
    g[0] = x[0] - theta[0];
    return 0.5 * g[0] * g[0];
}

static int jprox(const real_t* x, real_t gamma, const real_t* theta, const real_t* var, real_t* j)
{
    (void)x; (void)gamma; (void)theta; (void)var;
    j[0] = 1.0;
    return SUCCESS;
}

static int hvp(const real_t* v, const real_t* x, const real_t* theta, const real_t* var, real_t* h)
{
    (void)x; (void)theta; (void)var;
    h[0] = v[0];
    return SUCCESS;
}

static int loss(const real_t* x, const real_t* theta, const real_t* var, real_t* value, real_t* g)
{
    (void)theta; (void)var;
    *value = x[0];
    g[0] = 1.0;
    return SUCCESS;
}

static int vjp(const real_t* v, const real_t* x, const real_t* theta, const real_t* var, real_t* g)
{
    (void)x; (void)theta; (void)var;
    g[0] = -v[0];
    return SUCCESS;
}

static int constraint(const real_t* x, const real_t* theta, const real_t* var, real_t* c)
{
    (void)theta; (void)var;
    c[0] = x[0];
    return SUCCESS;
}

static int jtprod(const real_t* x, const real_t* theta, const real_t* var, const real_t* m, real_t* g)
{
    (void)x; (void)theta; (void)var;
    g[0] = m[0];
    return SUCCESS;
}

static int constraint_hvp(const real_t* v, const real_t* x, const real_t* theta,
    const real_t* var, const real_t* m, const real_t* rho, const real_t* lower,
    const real_t* upper, real_t* h)
{
    (void)x; (void)theta; (void)var; (void)m; (void)lower; (void)upper;
    observed_penalty = rho[0];
    h[0] = rho[0] * v[0];
    return SUCCESS;
}

static int constraint_vjp(const real_t* v, const real_t* x, const real_t* theta,
    const real_t* var, const real_t* m, const real_t* rho, const real_t* lower,
    const real_t* upper, real_t* g)
{
    (void)v; (void)x; (void)theta; (void)var; (void)m;
    (void)rho; (void)lower; (void)upper;
    g[0] = 0.0;
    return SUCCESS;
}

int main(void)
{
    alm_problem problem;
    alm_parameters params;
    alm_info info;
    optimizer_solve_info backward;
    real_t bound = 0.0, theta = 1.0, var = 0.0;
    real_t x = 0.0, lambda = 0.0, gradient = 0.0;
    memset(&problem, 0, sizeof(problem));
    memset(&params, 0, sizeof(params));
    problem.oracle.n = problem.oracle.ntheta = problem.oracle.nvar = 1;
    problem.oracle.proxg = prox; problem.oracle.cost_gradient = cost;
    problem.oracle.jprox = jprox; problem.oracle.hvp = hvp;
    problem.oracle.loss_grad = loss; problem.oracle.vjp = vjp;
    problem.ncon = 1;
    problem.constraint_lower = problem.constraint_upper = &bound;
    problem.constraint = constraint; problem.constraint_jtprod = jtprod;
    problem.constraint_hvp = constraint_hvp; problem.constraint_vjp = constraint_vjp;
    problem.inner_solver_params.max_iterations = 100;
    problem.inner_solver_params.tolerance = 1e-12;
    problem.inner_solver_params.buffer_size = 5;
    problem.backward_params.enable = TRUE;
    problem.backward_params.max_iterations = 100;
    problem.backward_params.tolerance = 1e-12;
    problem.backward_params.force_solver = PANDA_BACKWARD_SOLVER_CG;
    params.max_iterations = 2; params.tolerance = 1e-12;
    params.initial_penalty = 1.0; params.penalty_update_factor = 5.0;
    params.sufficient_decrease_factor = 0.25; params.warm_start_inner = TRUE;

    alm_solve_with_backward(&problem, &params, &x, &lambda, &theta, &var,
        &info, &gradient, &backward);
    /* The second solved subproblem uses lambda=0.5 and rho=1.
     * A further ALM update would change rho to 5 without another solve. */
    if (fabs(x-0.25) > 1e-8 || fabs(lambda-0.5) > 1e-8 ||
        observed_penalty != 1.0 || alm_get_penalty(0) != 1.0 ||
        info.penalty != 1.0 || fabs(gradient-0.5) > 1e-10 ||
        backward.final_residual > 1e-10 || info.iterations != 2) {
        printf("Last-subproblem mismatch: x=%g lambda=%g rho=%g gradient=%g\n",
            x, lambda, observed_penalty, gradient);
        return 1;
    }
    puts("ALM last-subproblem test passed");
    return 0;
}
