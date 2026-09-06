#include "static_casadi_oracle.h"
#include "lapanda_generated_config.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define HORIZON 20
#define DT 0.12
#define WHEELBASE 0.33
#define STEER_LIMIT 0.7
#define EPOCHS 150
#define LR 1.0e-3

static void fill_variable(double* variable, const double* teacher)
{
    unsigned int i;
    variable[0] = -1.2;
    variable[1] = 0.0;
    variable[2] = 0.0;
    variable[3] = 1.2;
    variable[4] = 0.0;
    variable[5] = 0.0;
    for (i = 0; i < LAPANDA_N; ++i) {
        variable[6 + i] = teacher ? teacher[i] : 0.0;
    }
}

static void fill_theta(double* theta, const double* margins)
{
    theta[0] = 5.0;
    theta[1] = 0.2;
    theta[2] = 1.0e-2;
    theta[3] = 1.0e-2;
    theta[4] = 20.0;
    theta[5] = margins[0];
    theta[6] = margins[1];
    theta[7] = margins[2];
    theta[8] = margins[3];
}

static double clamp(double value, double lower, double upper)
{
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static double angle_error(double angle)
{
    return atan2(sin(angle), cos(angle));
}

static void fill_initial_guess(double* solution)
{
    unsigned int k;

    for (k = 0; k < HORIZON; ++k) {
        solution[2 * k] = 0.0;
        solution[2 * k + 1] = 0.0;
    }
}

static void configure_problem(
    alm_problem* problem,
    struct solver_parameters* solver_params,
    struct backward_parameters* backward_params,
    double* constraint_lower,
    double* constraint_upper,
    int enable_backward)
{
    unsigned int i;
    for (i = 0; i < LAPANDA_NCON; ++i) {
        constraint_lower[i] = -1.0e20;
        constraint_upper[i] = 0.0;
    }

    solver_params->max_iterations = 2000;
    solver_params->tolerance = 1.0e-3;
    solver_params->buffer_size = 10;
    solver_params->max_stable_iter = 80;
    solver_params->verbose = 0;

    backward_params->enable = enable_backward ? 1 : 0;
    backward_params->tolerance = 1.0e-3;
    backward_params->max_iterations = 200;

    lapanda_static_init_alm_problem(
        problem,
        constraint_lower,
        constraint_upper,
        solver_params,
        backward_params);
}

static void configure_alm(alm_parameters* params)
{
    params->max_iterations = 100;
    params->tolerance = 1.0e-5;
    params->initial_penalty = 10000.0;
    params->penalty_update_factor = 10.0;
    params->max_penalty = 0.0;
    params->sufficient_decrease_factor = 0.25;
    params->verbose = 0;
    params->warm_start_inner = 1;
}

static int solve_forward(
    const double* theta,
    const double* variable,
    double* solution,
    double* multipliers,
    alm_info* info)
{
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    alm_problem problem;
    alm_parameters params;
    double constraint_lower[LAPANDA_NCON];
    double constraint_upper[LAPANDA_NCON];
    configure_problem(&problem, &solver_params, &backward_params, constraint_lower, constraint_upper, 0);
    configure_alm(&params);
    return alm_solve(&problem, &params, solution, multipliers, theta, variable, info);
}

static int solve_with_backward(
    const double* theta,
    const double* variable,
    double* solution,
    double* multipliers,
    alm_info* info,
    double* grad_theta,
    optimizer_solve_info* backward_info)
{
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    alm_problem problem;
    alm_parameters params;
    double constraint_lower[LAPANDA_NCON];
    double constraint_upper[LAPANDA_NCON];
    configure_problem(&problem, &solver_params, &backward_params, constraint_lower, constraint_upper, 1);
    configure_alm(&params);
    return alm_solve_with_backward_and_penalty0(
        &problem,
        &params,
        solution,
        multipliers,
        NULL,
        theta,
        variable,
        info,
        grad_theta,
        backward_info);
}

static double imitation_loss(const double* solution, const double* teacher)
{
    double loss = 0.0;
    unsigned int i;
    for (i = 0; i < LAPANDA_N; ++i) {
        const double r = solution[i] - teacher[i];
        loss += 0.5 * r * r;
    }
    return loss;
}

static double constraint_maximum(const double* theta, const double* variable, const double* solution)
{
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    alm_problem problem;
    double constraint_lower[LAPANDA_NCON];
    double constraint_upper[LAPANDA_NCON];
    double constraint[LAPANDA_NCON];
    double out = -1.0e300;
    unsigned int i;
    configure_problem(&problem, &solver_params, &backward_params, constraint_lower, constraint_upper, 0);
    if (problem.constraint(solution, theta, variable, constraint) == SUCCESS) {
        for (i = 0; i < LAPANDA_NCON; ++i) {
            if (constraint[i] > out) {
                out = constraint[i];
            }
        }
    }
    return out;
}

static unsigned int inner_iterations_sum(void)
{
    unsigned int sum = 0;
    unsigned int i;
    for (i = 0; i < alm_get_inner_iterations_count(); ++i) {
        sum += alm_get_inner_iterations(i);
    }
    return sum;
}

static void normalized_margin_step(const double* grad_theta, double* step)
{
    const double g0 = grad_theta[5];
    const double g1 = grad_theta[6];
    const double g2 = grad_theta[7];
    const double norm = sqrt(g0 * g0 + g1 * g1 + g2 * g2);
    step[0] = 0.0;
    step[1] = 0.0;
    step[2] = 0.0;
    step[3] = 0.0;
    if (isfinite(norm) && norm > 0.0) {
        step[0] = LR * g0 / norm;
        step[1] = LR * g1 / norm;
        step[2] = LR * g2 / norm;
    }
}

int main(void)
{
    const double teacher_margins[4] = {0.120, 0.120, 0.010, 0.220};
    double margins[4] = {0.150, 0.150, 0.160, 0.220};
    const double scales[6] = {1.0, 0.5, 0.25, 0.1, 0.05, 0.01};
    double theta[LAPANDA_NTHETA];
    double variable[LAPANDA_NVAR];
    double teacher_theta[LAPANDA_NTHETA];
    double teacher_variable[LAPANDA_NVAR];
    double teacher_solution[LAPANDA_N];
    double solution[LAPANDA_N];
    double candidate_solution[LAPANDA_N];
    double multipliers[LAPANDA_NCON];
    double candidate_multipliers[LAPANDA_NCON];
    double teacher_multipliers[LAPANDA_NCON];
    alm_info info;
    alm_info candidate_info;
    alm_info teacher_info;
    optimizer_solve_info backward_info;
    double grad_theta[LAPANDA_NTHETA];
    double step[4];
    double initial_loss = 0.0;
    double final_loss = 0.0;
    double mean_forward = 0.0;
    double mean_backward = 0.0;
    unsigned int epoch;
    int status;

    fill_theta(teacher_theta, teacher_margins);
    fill_variable(teacher_variable, NULL);
    fill_initial_guess(teacher_solution);
    memset(teacher_multipliers, 0, sizeof(teacher_multipliers));
    status = solve_forward(teacher_theta, teacher_variable, teacher_solution, teacher_multipliers, &teacher_info);
    if (status < 0) {
        printf("teacher_status=%d\n", status);
        return 1;
    }

    fill_theta(theta, margins);
    fill_variable(variable, teacher_solution);
    fill_initial_guess(solution);
    memset(multipliers, 0, sizeof(multipliers));

    printf("epoch,loss,margin_left,margin_right,margin_bottom,margin_top,forward_time_sec,backward_time_sec,outer_iterations,inner_iterations_sum,backward_iterations,final_residual,constraint_max,grad_norm\n");
    for (epoch = 0; epoch <= EPOCHS; ++epoch) {
        fill_theta(theta, margins);
        fill_variable(variable, teacher_solution);
        memset(grad_theta, 0, sizeof(grad_theta));
        memset(multipliers, 0, sizeof(multipliers));
        status = solve_with_backward(theta, variable, solution, multipliers, &info, grad_theta, &backward_info);
        if (status < 0) {
            printf("solve_failed_epoch=%u status=%d\n", epoch, status);
            return 1;
        }

        final_loss = imitation_loss(solution, teacher_solution);
        if (epoch == 0) {
            initial_loss = final_loss;
        }
        mean_forward += info.forward_time_sec;
        mean_backward += info.backward_time_sec;
        normalized_margin_step(grad_theta, step);
        {
            const double grad_norm = sqrt(
                grad_theta[5] * grad_theta[5] +
                grad_theta[6] * grad_theta[6] +
                grad_theta[7] * grad_theta[7]);
            printf("%u,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%u,%u,%u,%.17g,%.17g,%.17g\n",
                epoch,
                final_loss,
                margins[0],
                margins[1],
                margins[2],
                margins[3],
                info.forward_time_sec,
                info.backward_time_sec,
                info.iterations,
                inner_iterations_sum(),
                backward_info.iterations,
                info.final_residual,
                constraint_maximum(theta, variable, solution),
                grad_norm);
        }

        if (epoch < EPOCHS) {
            double best_margins[4] = {margins[0], margins[1], margins[2], margins[3]};
            double best_loss = final_loss;
            int accepted = 0;
            unsigned int s;
            for (s = 0; s < 6; ++s) {
                double candidate_margins[4] = {
                    clamp(margins[0] - scales[s] * step[0], 1.0e-4, 0.5),
                    clamp(margins[1] - scales[s] * step[1], 1.0e-4, 0.5),
                    clamp(margins[2] - scales[s] * step[2], 1.0e-4, 0.5),
                    margins[3]
                };
                double candidate_theta[LAPANDA_NTHETA];
                fill_theta(candidate_theta, candidate_margins);
                memcpy(candidate_solution, solution, sizeof(candidate_solution));
                memset(candidate_multipliers, 0, sizeof(candidate_multipliers));
                status = solve_forward(candidate_theta, variable, candidate_solution, candidate_multipliers, &candidate_info);
                if (status >= 0) {
                    const double candidate_loss = imitation_loss(candidate_solution, teacher_solution);
                    if (candidate_loss <= best_loss) {
                        best_loss = candidate_loss;
                        best_margins[0] = candidate_margins[0];
                        best_margins[1] = candidate_margins[1];
                        best_margins[2] = candidate_margins[2];
                        best_margins[3] = candidate_margins[3];
                        memcpy(solution, candidate_solution, sizeof(solution));
                        accepted = 1;
                        break;
                    }
                }
            }
            if (accepted) {
                margins[0] = best_margins[0];
                margins[1] = best_margins[1];
                margins[2] = best_margins[2];
                margins[3] = best_margins[3];
            }
        }
    }

    mean_forward /= (double)(EPOCHS + 1);
    mean_backward /= (double)(EPOCHS + 1);
    printf("summary_initial_loss=%.17g\n", initial_loss);
    printf("summary_final_loss=%.17g\n", final_loss);
    printf("summary_final_margins=%.17g,%.17g,%.17g,%.17g\n", margins[0], margins[1], margins[2], margins[3]);
    printf("summary_mean_forward_time_sec=%.17g\n", mean_forward);
    printf("summary_mean_backward_time_sec=%.17g\n", mean_backward);
    return 0;
}


