#include "../globals/globals.h"
#include "../include/optimizer.h"

#include <math.h>
#include <stdio.h>

static real_t quadratic_cost_gradient(
    const real_t* u,
    const real_t* theta,
    const real_t* variable,
    real_t* grad
)
{
    real_t cost;
    unsigned int i;

    (void)variable;

    cost = 0.0;
    for (i = 0; i < 2; ++i) {
        grad[i] = u[i] - theta[i];
        cost += 0.5 * grad[i] * grad[i];
    }

    return cost;
}

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

static int outer_loss_grad(
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* L,
    real_t* grad_u_L
)
{
    unsigned int i;

    (void)theta;

    *L = 0.0;
    for (i = 0; i < 2; ++i) {
        grad_u_L[i] = u_star[i] - variable[i];
        *L += 0.5 * grad_u_L[i] * grad_u_L[i];
    }

    return SUCCESS;
}

static int quadratic_vjp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* grad_theta
)
{
    (void)u_star;
    (void)theta;
    (void)variable;

    grad_theta[0] = -v[0];
    grad_theta[1] = -v[1];
    return SUCCESS;
}

static real_t max_abs_error(const real_t* x, const real_t* ref, unsigned int n)
{
    real_t err;
    real_t err_i;
    unsigned int i;

    err = 0.0;
    for (i = 0; i < n; ++i) {
        err_i = fabs(x[i] - ref[i]);
        if (err_i > err) {
            err = err_i;
        }
    }

    return err;
}

int main(void)
{
    struct optimizer_problem problem;
    real_t theta[2];
    real_t target[2];
    real_t u_star[2];
    real_t grad_theta[2];
    real_t expected_grad[2];
    optimizer_solve_info forward_info;
    optimizer_solve_info backward_info;
    int iters;

    theta[0] = 1.0;
    theta[1] = -2.0;
    target[0] = 0.25;
    target[1] = -1.0;
    u_star[0] = theta[0];
    u_star[1] = theta[1];

    problem.oracle.n = 2;
    problem.oracle.ntheta = 2;
    problem.oracle.nvar = 2;
    problem.oracle.proxg = identity_prox;
    problem.oracle.cost_gradient = quadratic_cost_gradient;
    problem.oracle.jprox = free_jprox;
    problem.oracle.hvp = identity_hvp;
    problem.oracle.loss_grad = outer_loss_grad;
    problem.oracle.vjp = quadratic_vjp;

    problem.solver_params.max_iterations = 50;
    problem.solver_params.tolerance = 1e-4;
    problem.solver_params.buffer_size = 5;
    problem.solver_params.max_stable_iter = 0;
    problem.solver_params.verbose = FALSE;

    problem.backward_params.enable = TRUE;
    problem.backward_params.tolerance = 1e-4;
    problem.backward_params.max_iterations = 20;
    problem.backward_params.restart = 20;
    problem.backward_params.force_solver = PANDA_BACKWARD_SOLVER_AUTO;
    problem.trace = NULL;
    problem.trace_context = NULL;

    if (optimizer_init(&problem) == FAILURE) {
        printf("optimizer_init failed\n");
        return 1;
    }

    iters = solve_problem(u_star, theta, target);
    (void)iters;

    if (solve_backward(u_star, grad_theta) == FAILURE) {
        printf("solve_backward failed\n");
        optimizer_cleanup();
        return 1;
    }

    if (optimizer_get_forward_info(&forward_info) == FAILURE ||
        optimizer_get_backward_info(&backward_info) == FAILURE) {
        printf("failed to get solve info\n");
        optimizer_cleanup();
        return 1;
    }

    if (backward_info.final_residual > problem.backward_params.tolerance) {
        printf("backward residual too large: %.16e\n",
               (double)backward_info.final_residual);
        optimizer_cleanup();
        return 1;
    }

    expected_grad[0] = theta[0] - target[0];
    expected_grad[1] = theta[1] - target[1];

    if (max_abs_error(u_star, theta, 2) > 1e-6 ||
        max_abs_error(grad_theta, expected_grad, 2) > 1e-6) {
        printf("quadratic test failed: u=[%.16e %.16e], grad=[%.16e %.16e]\n",
               (double)u_star[0],
               (double)u_star[1],
               (double)grad_theta[0],
               (double)grad_theta[1]);
        optimizer_cleanup();
        return 1;
    }

    optimizer_cleanup();
    printf("quadratic forward/backward test passed\n");
    return 0;
}
