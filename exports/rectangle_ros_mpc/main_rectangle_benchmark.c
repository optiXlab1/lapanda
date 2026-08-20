#include "static_casadi_oracle.h"
#include "LAPANDA_generated_config.h"

#include <stdio.h>
#include <string.h>

static void fill_problem_data(double* theta, double* variable)
{
    const double theta_value[LAPANDA_NTHETA] = {
        5.0, 0.2, 1e-2, 1e-2, 20.0, 0.10, 0.10, 0.10, 0.22
    };
    const double variable_value[LAPANDA_NVAR] = {
        -1.2, 0.0, 0.0, 1.2, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    unsigned int i;
    for (i = 0; i < LAPANDA_NTHETA; ++i) {
        theta[i] = theta_value[i];
    }
    for (i = 0; i < LAPANDA_NVAR; ++i) {
        variable[i] = variable_value[i];
    }
}

int main(void)
{
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    alm_problem problem;
    alm_parameters params;
    alm_info info;
    optimizer_solve_info backward_info;
    double solution[LAPANDA_N];
    double theta[LAPANDA_NTHETA];
    double variable[LAPANDA_NVAR > 0 ? LAPANDA_NVAR : 1];
    double multipliers[LAPANDA_NCON];
    double constraint_lower[LAPANDA_NCON];
    double constraint_upper[LAPANDA_NCON];
    double grad_theta[LAPANDA_NTHETA];
    double constraint[LAPANDA_NCON];
    double constraint_max;
    unsigned int i;
    unsigned int inner_sum;
    int status;

    memset(solution, 0, sizeof(solution));
    memset(multipliers, 0, sizeof(multipliers));
    memset(grad_theta, 0, sizeof(grad_theta));
    fill_problem_data(theta, variable);

    for (i = 0; i < LAPANDA_NCON; ++i) {
        constraint_lower[i] = -1.0e20;
        constraint_upper[i] = 0.0;
    }

    solver_params.max_iterations = 2000;
    solver_params.tolerance = 1e-3;
    solver_params.buffer_size = 10;
    solver_params.max_stable_iter = 80;
    solver_params.verbose = 0;

    backward_params.enable = 1;
    backward_params.tolerance = 1e-3;
    backward_params.max_iterations = 200;

    LAPANDA_static_init_alm_problem(
        &problem,
        constraint_lower,
        constraint_upper,
        &solver_params,
        &backward_params);

    params.max_iterations = 100;
    params.tolerance = 1e-4;
    params.initial_penalty = 10000.0;
    params.penalty_update_factor = 10.0;
    params.max_penalty = 0.0;
    params.sufficient_decrease_factor = 0.25;
    params.verbose = 0;
    params.warm_start_inner = 1;

    status = alm_solve_with_backward_and_penalty0(
        &problem,
        &params,
        solution,
        multipliers,
        NULL,
        theta,
        variable,
        &info,
        grad_theta,
        &backward_info);

    if (status < 0) {
        printf("status=%d\n", status);
        return 1;
    }

    constraint_max = -1.0e300;
    if (problem.constraint(solution, theta, variable, constraint) == SUCCESS) {
        for (i = 0; i < LAPANDA_NCON; ++i) {
            if (constraint[i] > constraint_max) {
                constraint_max = constraint[i];
            }
        }
    }

    inner_sum = 0;
    for (i = 0; i < alm_get_inner_iterations_count(); ++i) {
        inner_sum += alm_get_inner_iterations(i);
    }

    printf("status=%d\n", status);
    printf("outer_iterations=%u\n", info.iterations);
    printf("inner_iterations_sum=%u\n", inner_sum);
    printf("final_residual=%.17g\n", info.final_residual);
    printf("penalty=%.17g\n", info.penalty);
    printf("constraint_max=%.17g\n", constraint_max);
    printf("forward_time_sec=%.17g\n", info.forward_time_sec);
    printf("backward_time_sec=%.17g\n", info.backward_time_sec);
    printf("backward_iterations=%u\n", backward_info.iterations);
    printf("backward_residual=%.17g\n", backward_info.final_residual);
    printf("solution_0=%.17g\n", solution[0]);
    printf("solution_1=%.17g\n", solution[1]);
    return 0;
}
