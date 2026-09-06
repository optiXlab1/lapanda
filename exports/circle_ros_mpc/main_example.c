#include "static_casadi_oracle.h"
#include "lapanda_generated_config.h"

#include <stdio.h>
#include <string.h>

int main(void)
{
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    double solution[LAPANDA_N];
    double theta[LAPANDA_NTHETA];
    double variable[LAPANDA_NVAR > 0 ? LAPANDA_NVAR : 1];
    unsigned int i;

    memset(solution, 0, sizeof(solution));
    memset(theta, 0, sizeof(theta));
    memset(variable, 0, sizeof(variable));

    solver_params.max_iterations = 1000;
    solver_params.tolerance = 1e-4;
    solver_params.buffer_size = 10;
    solver_params.max_stable_iter = 80;
    solver_params.verbose = 0;

    backward_params.enable = 0;
    backward_params.tolerance = 1e-8;
    backward_params.max_iterations = 800;

#if LAPANDA_NCON > 0
    {
        alm_problem problem;
        alm_parameters params;
        alm_info info;
        double multipliers[LAPANDA_NCON];
        double constraint_lower[LAPANDA_NCON];
        double constraint_upper[LAPANDA_NCON];

        memset(multipliers, 0, sizeof(multipliers));
        for (i = 0; i < LAPANDA_NCON; ++i) {
            constraint_lower[i] = 0.0;
            constraint_upper[i] = 0.0;
        }

        lapanda_static_init_alm_problem(
            &problem,
            constraint_lower,
            constraint_upper,
            &solver_params,
            &backward_params);

        params.max_iterations = 50;
        params.tolerance = 1e-4;
        params.initial_penalty = 1.0;
        params.penalty_update_factor = 2.0;
        params.max_penalty = 0.0;
        params.sufficient_decrease_factor = 0.25;
        params.verbose = 0;

        (void)alm_solve(&problem, &params, solution, multipliers, theta, variable, &info);
        printf("ALM iterations: %u, residual: %.6e\n", info.iterations, info.final_residual);
    }
#else
    {
        struct optimizer_problem problem;
        optimizer_solve_info info;

        lapanda_static_init_panda_problem(&problem, &solver_params, &backward_params);
        (void)optimizer_init(&problem);
        (void)solve_problem(solution, theta, variable);
        (void)optimizer_get_forward_info(&info);
        (void)optimizer_cleanup();
        printf("PANDA iterations: %u, residual: %.6e\n", info.iterations, info.final_residual);
    }
#endif

    for (i = 0; i < LAPANDA_N; ++i) {
        printf("solution[%u] = %.16e\n", i, solution[i]);
    }
    return 0;
}
