#include "../globals/globals.h"
#include "function_evaluator.h"
#include "panda_forward.h"

#include <math.h>
#include <stdlib.h>

#include "../include/optimizer.h"

static struct optimizer_problem* problem; /* data related to the problem */
static unsigned char initialized = FALSE;
static real_t g_final_gamma;
static optimizer_solve_info g_forward_info;
static optimizer_solve_info g_backward_info;

static void save_solution(real_t* solution);
static void save_gamma(real_t t_gamma);
int solve_problem(
    real_t* solution,
    const real_t* theta,
    const real_t* variable
)
{
    unsigned int i_panda;
    real_t current_residual;
    int status;

    if (initialized == FALSE) {
        return FAILURE;
    }

    g_forward_info.iterations = 0;
    g_forward_info.final_residual = 0.0;

    function_evaluator_set_parameters(theta, variable);

    status = panda_forward_set_initial(solution);
    if (status == FAILURE) {
        return FAILURE;
    }

    current_residual = problem->solver_params.tolerance * 10.0;

    for (i_panda = 0; i_panda < problem->solver_params.max_iterations; i_panda++) {
        if (current_residual < problem->solver_params.tolerance) {
            break;
        }

        current_residual = panda_forward_step();

        if (problem->trace != NULL) {
            problem->trace(problem->trace_context, i_panda, current_residual);
        }

        if (current_residual > MACHINE_ACCURACY) {
            save_solution(solution);
        }
    }

    save_gamma(panda_forward_get_gamma());
    g_forward_info.iterations = i_panda;
    g_forward_info.final_residual = current_residual;
    panda_forward_reset();

    return i_panda;
}

static void save_gamma(real_t t_gamma)
{
    g_final_gamma = t_gamma;
}

static void save_solution(real_t* solution)
{
    unsigned int i;
    const real_t* accepted_solution = panda_forward_get_solution();

    for (i = 0; i < problem->oracle.n; i++) {
        solution[i] = accepted_solution[i];
    }
}

real_t get_gamma()
{
    return g_final_gamma;
}

int optimizer_init(struct optimizer_problem* problem_)
{
    panda_params params;

    if (initialized == TRUE) {
        optimizer_cleanup();
    }

    problem = problem_;

    if (function_evaluator_init(problem) == FAILURE) {
        return FAILURE;
    }

    g_forward_info.iterations = 0;
    g_forward_info.final_residual = 0.0;
    g_backward_info.iterations = 0;
    g_backward_info.final_residual = 0.0;

    params.n = problem->oracle.n;
    params.alpha = 1.0 - PROXIMAL_GRAD_DESC_SAFETY_VALUE;
    params.beta = 0.5;
    params.minimum_gamma = 1e-6;
    params.max_backtracks = FBE_LINESEARCH_MAX_ITERATIONS;
    params.lbfgs_memory = problem->solver_params.buffer_size;
    params.enable_gamma_enlarge =
        (problem->solver_params.max_stable_iter > 0) ? TRUE : FALSE;
    params.max_stable_iter = problem->solver_params.max_stable_iter;
    params.residual_enlarge_threshold = 1e-3;
    params.gamma_enlarge_factor = 2.0;

    if (panda_forward_init(&params) == FAILURE) {
        function_evaluator_cleanup();
        return FAILURE;
    }

    initialized = TRUE;
    return SUCCESS;
}

int optimizer_init_with_custom_constraint(struct optimizer_problem* problem_,
                                          panda_prox_fun proxg)
{
    problem_->oracle.proxg = proxg;
    return optimizer_init(problem_);
}

int solve_backward(
    const real_t* u_star,
    real_t* dLdtheta
)
{
    return solve_backward_with_options(
        u_star,
        NULL,
        dLdtheta);
}

int solve_backward_with_options(
    const real_t* u_star,
    const panda_backward_options* opts,
    real_t* dLdtheta
)
{
    panda_backward_options default_opts;
    const panda_backward_options* used_opts;

    if (initialized == FALSE) {
        return FAILURE;
    }

    if (problem->backward_params.enable == FALSE) {
        return FAILURE;
    }

    g_backward_info.iterations = 0;
    g_backward_info.final_residual = 0.0;

    used_opts = opts;
    if (used_opts == NULL) {
        default_opts.enable = TRUE;
        default_opts.tol = problem->backward_params.tolerance;
        default_opts.max_iter = problem->backward_params.max_iterations;
        default_opts.restart = problem->backward_params.restart;
        default_opts.sym_tol = 1e-8;
        default_opts.sym_num_tests = 5;
        default_opts.force_solver = problem->backward_params.force_solver;
        default_opts.recover_active = TRUE;

        if (default_opts.tol <= 0.0) {
            default_opts.tol = 1e-4;
        }
        if (default_opts.max_iter == 0) {
            default_opts.max_iter = 200;
        }
        if (default_opts.restart == 0 ||
            default_opts.restart > default_opts.max_iter) {
            default_opts.restart = 40;
        }
        if (default_opts.force_solver < PANDA_BACKWARD_SOLVER_AUTO ||
            default_opts.force_solver > PANDA_BACKWARD_SOLVER_CG) {
            default_opts.force_solver = PANDA_BACKWARD_SOLVER_AUTO;
        }

        used_opts = &default_opts;
    }

    return panda_backward_compute(
        used_opts,
        u_star,
        g_final_gamma,
        dLdtheta,
        &g_backward_info.final_residual,
        &g_backward_info.iterations);
}

int optimizer_cleanup(void)
{
    panda_forward_cleanup();
    function_evaluator_cleanup();
    initialized = FALSE;
    return SUCCESS;
}

int optimizer_get_forward_info(optimizer_solve_info* info)
{
    if (info == NULL) {
        return FAILURE;
    }

    *info = g_forward_info;
    return SUCCESS;
}

int optimizer_get_backward_info(optimizer_solve_info* info)
{
    if (info == NULL) {
        return FAILURE;
    }

    *info = g_backward_info;
    return SUCCESS;
}
