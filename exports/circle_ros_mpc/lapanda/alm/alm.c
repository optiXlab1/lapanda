#include "alm.h"

#include "../panda/panda_forward.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

typedef struct {
    const alm_problem* problem;
    const real_t* multipliers;
    const real_t* penalties;
    real_t* constraint;
    real_t* slack;
    real_t* residual;
    real_t* weight;
    real_t* gradient;
} alm_context;

static alm_context g_ctx;

typedef struct {
    FILE* fp;
    unsigned int outer_iter;
} alm_trace_context;

#define ALM_MAX_RECORDED_INNER_ITERATIONS 1024

static unsigned int g_inner_iterations[ALM_MAX_RECORDED_INNER_ITERATIONS];
static unsigned int g_inner_iterations_count = 0;
static real_t* g_last_penalties = NULL;
static unsigned int g_last_penalties_count = 0;

static real_t alm_now_sec(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (QueryPerformanceFrequency(&frequency) && QueryPerformanceCounter(&counter)) {
        return (real_t)counter.QuadPart / (real_t)frequency.QuadPart;
    }
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (real_t)ts.tv_sec + (real_t)ts.tv_nsec * (real_t)1e-9;
    }
#elif defined(TIME_UTC)
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
        return (real_t)ts.tv_sec + (real_t)ts.tv_nsec * (real_t)1e-9;
    }
#endif
    return (real_t)clock() / (real_t)CLOCKS_PER_SEC;
}

static int alm_solve_internal(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info
);

static real_t project_box(real_t x, real_t lb, real_t ub)
{
    if (x < lb) {
        return lb;
    }
    if (x > ub) {
        return ub;
    }
    return x;
}

static void alm_inner_trace(
    void* context,
    unsigned int inner_iter,
    real_t residual
)
{
    alm_trace_context* trace_ctx;

    trace_ctx = (alm_trace_context*)context;
    if (trace_ctx == NULL || trace_ctx->fp == NULL) {
        return;
    }

    fprintf(trace_ctx->fp,
            "inner,%u,%u,%.16e,,,"
            "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%u\n",
            trace_ctx->outer_iter,
            inner_iter,
            (double)residual,
            (double)panda_forward_get_gamma(),
            (double)panda_forward_get_tau(),
            (double)panda_forward_get_phi(),
            (double)panda_forward_get_f_x(),
            (double)panda_forward_get_f_z(),
            (double)panda_forward_get_g_z(),
            (double)panda_forward_get_residual_norm2(),
            (unsigned int)panda_forward_get_upper_ok());
    fflush(trace_ctx->fp);
}

static real_t alm_cost_gradient(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    real_t* output_gradient
)
{
    real_t value;
    real_t alm_value;
    real_t shifted;
    unsigned int i;

    value = g_ctx.problem->oracle.cost_gradient(
        input,
        theta,
        variable,
        output_gradient);

    if (g_ctx.problem->constraint(
            input,
            theta,
            variable,
            g_ctx.constraint) == FAILURE) {
        return value;
    }

    alm_value = 0.0;
    for (i = 0; i < g_ctx.problem->ncon; ++i) {
        shifted = g_ctx.constraint[i]
            + g_ctx.multipliers[i] / g_ctx.penalties[i];
        g_ctx.slack[i] = project_box(
            shifted,
            g_ctx.problem->constraint_lower[i],
            g_ctx.problem->constraint_upper[i]);
        g_ctx.residual[i] = g_ctx.constraint[i] - g_ctx.slack[i];
        g_ctx.weight[i] =
            g_ctx.multipliers[i] + g_ctx.penalties[i] * g_ctx.residual[i];
        alm_value += g_ctx.multipliers[i] * g_ctx.residual[i]
            + 0.5 * g_ctx.penalties[i]
            * g_ctx.residual[i] * g_ctx.residual[i];
    }

    if (g_ctx.problem->constraint_jtprod(
            input,
            theta,
            variable,
            g_ctx.weight,
            g_ctx.gradient) == SUCCESS) {
        for (i = 0; i < g_ctx.problem->oracle.n; ++i) {
            output_gradient[i] += g_ctx.gradient[i];
        }
    }

    return value + alm_value;
}

static real_t alm_proxg(
    real_t* input,
    real_t gamma,
    const real_t* theta,
    const real_t* variable
)
{
    return g_ctx.problem->oracle.proxg(input, gamma, theta, variable);
}

static int alm_jprox(
    const real_t* u_star,
    real_t gamma,
    const real_t* theta,
    const real_t* variable,
    real_t* J
)
{
    return g_ctx.problem->oracle.jprox(u_star, gamma, theta, variable, J);
}

static int alm_hvp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* Hv
)
{
    real_t* penalty_Hv;
    unsigned int i;

    if (g_ctx.problem->oracle.hvp(
            v,
            u_star,
            theta,
            variable,
            Hv) == FAILURE) {
        return FAILURE;
    }

    if (g_ctx.penalties == NULL) {
        return SUCCESS;
    }
    if (g_ctx.problem->constraint_hvp == NULL) {
        return FAILURE;
    }

    penalty_Hv = (real_t*)malloc(sizeof(real_t) * g_ctx.problem->oracle.n);
    if (penalty_Hv == NULL) {
        return FAILURE;
    }

    if (g_ctx.problem->constraint_hvp(
            v,
            u_star,
            theta,
            variable,
            g_ctx.multipliers,
            g_ctx.penalties,
            g_ctx.problem->constraint_lower,
            g_ctx.problem->constraint_upper,
            penalty_Hv) == FAILURE) {
        free(penalty_Hv);
        return FAILURE;
    }

    for (i = 0; i < g_ctx.problem->oracle.n; ++i) {
        Hv[i] += penalty_Hv[i];
    }

    free(penalty_Hv);
    return SUCCESS;
}

static int alm_loss_grad(
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* L,
    real_t* grad_u_L
)
{
    return g_ctx.problem->oracle.loss_grad(
        u_star,
        theta,
        variable,
        L,
        grad_u_L);
}

static int alm_vjp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* grad_theta
)
{
    real_t* penalty_grad_theta;
    unsigned int i;

    if (g_ctx.problem->oracle.vjp(
            v,
            u_star,
            theta,
            variable,
            grad_theta) == FAILURE) {
        return FAILURE;
    }

    if (g_ctx.penalties == NULL) {
        return SUCCESS;
    }

    if (g_ctx.problem->constraint_vjp == NULL) {
        return FAILURE;
    }

    penalty_grad_theta =
        (real_t*)malloc(sizeof(real_t) * g_ctx.problem->oracle.ntheta);
    if (penalty_grad_theta == NULL) {
        return FAILURE;
    }

    if (g_ctx.problem->constraint_vjp(
            v,
            u_star,
            theta,
            variable,
            g_ctx.multipliers,
            g_ctx.penalties,
            g_ctx.problem->constraint_lower,
            g_ctx.problem->constraint_upper,
            penalty_grad_theta) == FAILURE) {
        free(penalty_grad_theta);
        return FAILURE;
    }

    for (i = 0; i < g_ctx.problem->oracle.ntheta; ++i) {
        grad_theta[i] += penalty_grad_theta[i];
    }

    free(penalty_grad_theta);
    return SUCCESS;
}

static real_t compute_residual_inf_norm(
    const alm_problem* problem,
    const real_t* solution,
    const real_t* theta,
    const real_t* variable,
    const real_t* multipliers,
    const real_t* penalties,
    real_t* constraint,
    real_t* slack,
    real_t* residual
)
{
    real_t norm_inf;
    real_t abs_i;
    real_t shifted;
    unsigned int i;

    if (problem->constraint(
            solution,
            theta,
            variable,
            constraint) == FAILURE) {
        return LARGE;
    }

    norm_inf = 0.0;
    for (i = 0; i < problem->ncon; ++i) {
        shifted = constraint[i] + multipliers[i] / penalties[i];
        slack[i] = project_box(
            shifted,
            problem->constraint_lower[i],
            problem->constraint_upper[i]);
        residual[i] = constraint[i] - slack[i];
        abs_i = fabs(residual[i]);
        if (abs_i > norm_inf) {
            norm_inf = abs_i;
        }
    }

    return norm_inf;
}

static void fill_default_params(
    const alm_parameters* params,
    alm_parameters* out
)
{
    if (params != NULL) {
        *out = *params;
    } else {
        out->max_iterations = 10;
        out->tolerance = 1e-6;
        out->initial_penalty = 10.0;
        out->penalty_update_factor = 10.0;
        out->max_penalty = 0.0;
        out->sufficient_decrease_factor = 0.25;
        out->verbose = FALSE;
        out->warm_start_inner = TRUE;
    }

    if (out->max_iterations == 0) {
        out->max_iterations = 10;
    }
    if (out->tolerance <= 0.0) {
        out->tolerance = 1e-6;
    }
    if (out->initial_penalty <= 0.0) {
        out->initial_penalty = 10.0;
    }
    if (out->penalty_update_factor <= 1.0) {
        out->penalty_update_factor = 10.0;
    }
    if (out->max_penalty > 0.0 && out->max_penalty < out->initial_penalty) {
        out->max_penalty = out->initial_penalty;
    }
    if (out->sufficient_decrease_factor <= 0.0 ||
        out->sufficient_decrease_factor >= 1.0) {
        out->sufficient_decrease_factor = 0.25;
    }
    if (out->warm_start_inner != FALSE && out->warm_start_inner != TRUE) {
        out->warm_start_inner = TRUE;
    }
}

static void clear_context(void)
{
    free(g_ctx.constraint);
    free(g_ctx.slack);
    free(g_ctx.residual);
    free(g_ctx.weight);
    free(g_ctx.gradient);

    g_ctx.problem = NULL;
    g_ctx.multipliers = NULL;
    g_ctx.penalties = NULL;
    g_ctx.constraint = NULL;
    g_ctx.slack = NULL;
    g_ctx.residual = NULL;
    g_ctx.weight = NULL;
    g_ctx.gradient = NULL;
}

unsigned int alm_get_inner_iterations_count(void)
{
    return g_inner_iterations_count;
}

unsigned int alm_get_inner_iterations(unsigned int index)
{
    if (index >= g_inner_iterations_count) {
        return 0;
    }
    return g_inner_iterations[index];
}

unsigned int alm_get_penalties_count(void)
{
    return g_last_penalties_count;
}

real_t alm_get_penalty(unsigned int index)
{
    if (index >= g_last_penalties_count || g_last_penalties == NULL) {
        return 0.0;
    }
    return g_last_penalties[index];
}

int alm_solve(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    const real_t* theta,
    const real_t* variable,
    alm_info* info
)
{
    return alm_solve_internal(
        problem,
        params,
        solution,
        multipliers,
        NULL,
        theta,
        variable,
        info,
        NULL,
        NULL);
}

int alm_solve_with_backward(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info
)
{
    return alm_solve_internal(
        problem,
        params,
        solution,
        multipliers,
        NULL,
        theta,
        variable,
        info,
        dLdtheta,
        backward_info);
}

int alm_solve_with_penalty0(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info
)
{
    return alm_solve_internal(
        problem,
        params,
        solution,
        multipliers,
        penalty0,
        theta,
        variable,
        info,
        NULL,
        NULL);
}

int alm_solve_with_backward_and_penalty0(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info
)
{
    return alm_solve_internal(
        problem,
        params,
        solution,
        multipliers,
        penalty0,
        theta,
        variable,
        info,
        dLdtheta,
        backward_info);
}

static int alm_solve_internal(
    const alm_problem* problem,
    const alm_parameters* params,
    real_t* solution,
    real_t* multipliers,
    real_t* penalty0,
    const real_t* theta,
    const real_t* variable,
    alm_info* info,
    real_t* dLdtheta,
    optimizer_solve_info* backward_info
)
{
    const int backward_failure = -1;
    struct optimizer_problem inner_problem;
    alm_parameters used_params;
    real_t previous_violation;
    real_t violation;
    real_t factor;
    real_t max_penalty;
    real_t* penalties;
    real_t* previous_residual;
    real_t* initial_solution;
    unsigned int i;
    unsigned int k;
    int status;
    int inner_status;
    optimizer_solve_info inner_info;
    FILE* inner_log_fp;
    alm_trace_context trace_ctx;

    penalties = NULL;
    previous_residual = NULL;
    initial_solution = NULL;
    inner_log_fp = NULL;
    trace_ctx.fp = NULL;
    trace_ctx.outer_iter = 0;
    g_inner_iterations_count = 0;
    free(g_last_penalties);
    g_last_penalties = NULL;
    g_last_penalties_count = 0;

    if (problem == NULL || solution == NULL || multipliers == NULL) {
        return FAILURE;
    }
    if (problem->ncon == 0 ||
        problem->constraint == NULL ||
        problem->constraint_jtprod == NULL ||
        problem->constraint_lower == NULL ||
        problem->constraint_upper == NULL) {
        return FAILURE;
    }

    g_ctx.problem = problem;
    g_ctx.multipliers = multipliers;
    g_ctx.penalties = NULL;
    g_ctx.constraint = NULL;
    g_ctx.slack = NULL;
    g_ctx.residual = NULL;
    g_ctx.weight = NULL;
    g_ctx.gradient = NULL;

    g_ctx.constraint = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (g_ctx.constraint == NULL) goto fail;
    g_ctx.slack = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (g_ctx.slack == NULL) goto fail;
    g_ctx.residual = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (g_ctx.residual == NULL) goto fail;
    g_ctx.weight = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (g_ctx.weight == NULL) goto fail;
    g_ctx.gradient = (real_t*)malloc(sizeof(real_t) * problem->oracle.n);
    if (g_ctx.gradient == NULL) goto fail;
    penalties = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (penalties == NULL) goto fail;
    previous_residual = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (previous_residual == NULL) goto fail;
    initial_solution = (real_t*)malloc(sizeof(real_t) * problem->oracle.n);
    if (initial_solution == NULL) goto fail;

    fill_default_params(params, &used_params);

    for (i = 0; i < problem->oracle.n; ++i) {
        initial_solution[i] = solution[i];
    }

    for (i = 0; i < problem->ncon; ++i) {
        penalties[i] = (penalty0 != NULL && penalty0[i] > 0.0)
            ? penalty0[i]
            : used_params.initial_penalty;
        previous_residual[i] = LARGE;
    }

    inner_problem.oracle = problem->oracle;
    inner_problem.oracle.cost_gradient = alm_cost_gradient;
    inner_problem.oracle.proxg = alm_proxg;
    inner_problem.oracle.jprox = alm_jprox;
    inner_problem.oracle.hvp = alm_hvp;
    inner_problem.oracle.loss_grad = alm_loss_grad;
    inner_problem.oracle.vjp = alm_vjp;
    inner_problem.solver_params = problem->inner_solver_params;
    inner_problem.solver_params.verbose = FALSE;
    inner_problem.backward_params = problem->backward_params;
    inner_problem.trace = NULL;
    inner_problem.trace_context = NULL;
    if (dLdtheta == NULL) {
        inner_problem.backward_params.enable = FALSE;
    }

    if (optimizer_init(&inner_problem) == FAILURE) {
        goto fail;
    }

    if (used_params.verbose != FALSE) {
        inner_log_fp = fopen("LAPANDA_log.csv", "w");
        if (inner_log_fp != NULL) {
            fprintf(inner_log_fp,
                    "event,outer_iter,inner_iter,residual,alm_violation,penalty,"
                    "gamma,tau,phi,f_x,f_z,g_z,residual_norm2,upper_ok\n");
            fflush(inner_log_fp);
            trace_ctx.fp = inner_log_fp;
            inner_problem.trace = alm_inner_trace;
            inner_problem.trace_context = &trace_ctx;
        }
    }

    previous_violation = LARGE;
    status = FAILURE;

    if (info != NULL) {
        info->iterations = 0;
        info->final_residual = LARGE;
        info->penalty = used_params.initial_penalty;
        info->forward_time_sec = 0.0;
        info->backward_time_sec = 0.0;
    }

    const real_t forward_t0 = alm_now_sec();
    for (k = 0; k < used_params.max_iterations; ++k) {
        g_ctx.penalties = penalties;
        trace_ctx.outer_iter = k + 1;
        if (used_params.warm_start_inner == FALSE && k > 0) {
            for (i = 0; i < problem->oracle.n; ++i) {
                solution[i] = initial_solution[i];
            }
        }

        inner_status = solve_problem(solution, theta, variable);
        (void)optimizer_get_forward_info(&inner_info);
        if (g_inner_iterations_count < ALM_MAX_RECORDED_INNER_ITERATIONS) {
            g_inner_iterations[g_inner_iterations_count] = inner_info.iterations;
            g_inner_iterations_count++;
        }
        if (inner_status == FAILURE &&
            inner_info.iterations == 0 &&
            inner_info.final_residual == 0.0) {
            optimizer_cleanup();
            goto fail;
        }

        violation = compute_residual_inf_norm(
            problem,
            solution,
            theta,
            variable,
            multipliers,
            penalties,
            g_ctx.constraint,
            g_ctx.slack,
            g_ctx.residual);

        max_penalty = penalties[0];
        for (i = 1; i < problem->ncon; ++i) {
            if (penalties[i] > max_penalty) {
                max_penalty = penalties[i];
            }
        }

        if (info != NULL) {
            info->iterations = k + 1;
            info->final_residual = violation;
            info->penalty = max_penalty;
        }

        if (inner_log_fp != NULL) {
            fprintf(inner_log_fp,
                    "outer,%u,%u,%.16e,%.16e,%.16e,"
                    "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%u\n",
                    k + 1,
                    inner_info.iterations,
                    (double)inner_info.final_residual,
                    (double)violation,
                    (double)max_penalty,
                    (double)panda_forward_get_gamma(),
                    (double)panda_forward_get_tau(),
                    (double)panda_forward_get_phi(),
                    (double)panda_forward_get_f_x(),
                    (double)panda_forward_get_f_z(),
                    (double)panda_forward_get_g_z(),
                    (double)panda_forward_get_residual_norm2(),
                    (unsigned int)panda_forward_get_upper_ok());
            fflush(inner_log_fp);
        }

        if (violation <= used_params.tolerance) {
            status = k + 1;
            break;
        }

        for (i = 0; i < problem->ncon; ++i) {
            multipliers[i] += penalties[i] * g_ctx.residual[i];

            if (fabs(g_ctx.residual[i]) >
                used_params.sufficient_decrease_factor
                * fabs(previous_residual[i])) {
                factor = used_params.penalty_update_factor
                    * fabs(g_ctx.residual[i]) / violation;
                if (factor < 1.0) {
                    factor = 1.0;
                }
                penalties[i] *= factor;
                if (used_params.max_penalty > 0.0 &&
                    penalties[i] > used_params.max_penalty) {
                    penalties[i] = used_params.max_penalty;
                }
            }

            previous_residual[i] = g_ctx.residual[i];
        }

        previous_violation = violation;
        (void)previous_violation;
    }
    if (info != NULL) {
        info->forward_time_sec = alm_now_sec() - forward_t0;
    }

    if (status == FAILURE) {
        status = used_params.max_iterations;
    }

    g_last_penalties = (real_t*)malloc(sizeof(real_t) * problem->ncon);
    if (g_last_penalties != NULL) {
        for (i = 0; i < problem->ncon; ++i) {
            g_last_penalties[i] = penalties[i];
        }
        g_last_penalties_count = problem->ncon;
    }

    if (dLdtheta != NULL) {
        const real_t backward_t0 = alm_now_sec();
        if (solve_backward(solution, dLdtheta) == FAILURE) {
            optimizer_cleanup();
            if (inner_log_fp != NULL) {
                fclose(inner_log_fp);
            }
            clear_context();
            free(penalties);
            free(previous_residual);
            free(initial_solution);
            return backward_failure;
        }
        if (info != NULL) {
            info->backward_time_sec = alm_now_sec() - backward_t0;
        }
        if (backward_info != NULL) {
            (void)optimizer_get_backward_info(backward_info);
        }
    }

    optimizer_cleanup();
    if (inner_log_fp != NULL) {
        fclose(inner_log_fp);
    }
    clear_context();
    free(penalties);
    free(previous_residual);
    free(initial_solution);
    return status;

fail:
    if (inner_log_fp != NULL) {
        fclose(inner_log_fp);
    }
    clear_context();
    free(penalties);
    free(previous_residual);
    free(initial_solution);
    return FAILURE;
}
