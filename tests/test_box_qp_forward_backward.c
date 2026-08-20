#include "../globals/globals.h"
#include "../include/optimizer.h"

#include <math.h>
#include <stdio.h>

static const real_t g_lb[4] = {-1.0, -0.5, -2.0, 0.0};
static const real_t g_ub[4] = { 1.0,  0.5,  2.0, 1.5};

static real_t box_cost_gradient(
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
    for (i = 0; i < 4; ++i) {
        grad[i] = u[i] - theta[i];
        cost += 0.5 * grad[i] * grad[i];
    }

    return cost;
}

static real_t box_prox(
    real_t* input,
    real_t gamma,
    const real_t* theta,
    const real_t* variable
)
{
    unsigned int i;

    (void)gamma;
    (void)theta;
    (void)variable;

    for (i = 0; i < 4; ++i) {
        if (input[i] < g_lb[i]) {
            input[i] = g_lb[i];
        } else if (input[i] > g_ub[i]) {
            input[i] = g_ub[i];
        }
    }

    return 0.0;
}

static int box_jprox(
    const real_t* u_star,
    real_t gamma,
    const real_t* theta,
    const real_t* variable,
    real_t* J
)
{
    real_t active_tol;
    unsigned int i;

    (void)gamma;
    (void)theta;
    (void)variable;

    active_tol = 1e-10;
    for (i = 0; i < 4; ++i) {
        if (u_star[i] <= g_lb[i] + active_tol ||
            u_star[i] >= g_ub[i] - active_tol) {
            J[i] = 0.0;
        } else {
            J[i] = 1.0;
        }
    }

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
    unsigned int i;

    (void)u_star;
    (void)theta;
    (void)variable;

    for (i = 0; i < 4; ++i) {
        Hv[i] = v[i];
    }

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
    for (i = 0; i < 4; ++i) {
        grad_u_L[i] = u_star[i] - variable[i];
        *L += 0.5 * grad_u_L[i] * grad_u_L[i];
    }

    return SUCCESS;
}

static int box_vjp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* grad_theta
)
{
    unsigned int i;

    (void)u_star;
    (void)theta;
    (void)variable;

    /*
     * grad_u ell = u - theta, so
     * (d/dtheta grad_u ell)^T v = -v.
     */
    for (i = 0; i < 4; ++i) {
        grad_theta[i] = -v[i];
    }

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

static real_t clip(real_t x, real_t lb, real_t ub)
{
    if (x < lb) {
        return lb;
    }
    if (x > ub) {
        return ub;
    }
    return x;
}

int main(void)
{
    struct optimizer_problem problem;
    optimizer_solve_info backward_info;
    real_t theta[4];
    real_t target[4];
    real_t u_star[4];
    real_t expected_u[4];
    real_t grad_theta[4];
    real_t expected_grad[4];
    real_t J[4];
    unsigned int i;

    theta[0] = -2.0;  /* active lower */
    theta[1] = 0.25;  /* free */
    theta[2] = 3.0;   /* active upper */
    theta[3] = 0.75;  /* free */

    target[0] = -0.2;
    target[1] = -0.1;
    target[2] = 1.25;
    target[3] = 1.0;

    for (i = 0; i < 4; ++i) {
        expected_u[i] = clip(theta[i], g_lb[i], g_ub[i]);
        u_star[i] = expected_u[i];
    }

    problem.oracle.n = 4;
    problem.oracle.ntheta = 4;
    problem.oracle.nvar = 4;
    problem.oracle.proxg = box_prox;
    problem.oracle.cost_gradient = box_cost_gradient;
    problem.oracle.jprox = box_jprox;
    problem.oracle.hvp = identity_hvp;
    problem.oracle.loss_grad = outer_loss_grad;
    problem.oracle.vjp = box_vjp;

    problem.solver_params.max_iterations = 30;
    problem.solver_params.tolerance = 1e-9;
    problem.solver_params.buffer_size = 5;
    problem.solver_params.max_stable_iter = 0;
    problem.solver_params.verbose = FALSE;

    problem.backward_params.enable = TRUE;
    problem.backward_params.tolerance = 1e-12;
    problem.backward_params.max_iterations = 20;
    problem.trace = NULL;
    problem.trace_context = NULL;

    if (optimizer_init(&problem) == FAILURE) {
        printf("optimizer_init failed\n");
        return 1;
    }

    /*
     * Starting at the clipped solution makes this test focus on the active-set
     * backward formula rather than forward convergence behavior.
     */
    solve_problem(u_star, theta, target);

    if (solve_backward(u_star, grad_theta) == FAILURE) {
        printf("solve_backward failed\n");
        optimizer_cleanup();
        return 1;
    }

    if (box_jprox(u_star, get_gamma(), theta, target, J) == FAILURE) {
        optimizer_cleanup();
        return 1;
    }

    for (i = 0; i < 4; ++i) {
        expected_grad[i] = J[i] * (expected_u[i] - target[i]);
    }

    if (optimizer_get_backward_info(&backward_info) == FAILURE) {
        printf("optimizer_get_backward_info failed\n");
        optimizer_cleanup();
        return 1;
    }

    if (max_abs_error(u_star, expected_u, 4) > 1e-7 ||
        max_abs_error(grad_theta, expected_grad, 4) > 1e-7 ||
        backward_info.final_residual > problem.backward_params.tolerance) {
        printf("box QP test failed\n");
        printf("u      = [%.16e %.16e %.16e %.16e]\n",
               (double)u_star[0], (double)u_star[1],
               (double)u_star[2], (double)u_star[3]);
        printf("grad   = [%.16e %.16e %.16e %.16e]\n",
               (double)grad_theta[0], (double)grad_theta[1],
               (double)grad_theta[2], (double)grad_theta[3]);
        printf("expect = [%.16e %.16e %.16e %.16e]\n",
               (double)expected_grad[0], (double)expected_grad[1],
               (double)expected_grad[2], (double)expected_grad[3]);
        printf("backward residual = %.16e\n",
               (double)backward_info.final_residual);
        optimizer_cleanup();
        return 1;
    }

    optimizer_cleanup();
    printf("box QP forward/backward test passed\n");
    return 0;
}
