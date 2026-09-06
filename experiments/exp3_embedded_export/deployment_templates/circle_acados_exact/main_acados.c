#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_circle_forward.h"
#include "acados_solver_circle_sensitivity.h"

#define NX CIRCLE_FORWARD_NX
#define NU CIRCLE_FORWARD_NU
#define N  CIRCLE_FORWARD_N
#define NBX0 CIRCLE_FORWARD_NBX0
#define NP_GLOBAL CIRCLE_FORWARD_NP_GLOBAL

static double env_double(const char *name, double fallback)
{
    const char *value = getenv(name);
    char *end = NULL;
    double parsed;
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    parsed = strtod(value, &end);
    return (end != value) ? parsed : fallback;
}

static double wall_now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void override_tolerances(ocp_nlp_config *config, void *opts, double tol)
{
    if (tol > 0.0) {
        ocp_nlp_solver_opts_set(config, opts, "tol_stat", &tol);
        ocp_nlp_solver_opts_set(config, opts, "tol_eq", &tol);
        ocp_nlp_solver_opts_set(config, opts, "tol_ineq", &tol);
        ocp_nlp_solver_opts_set(config, opts, "tol_comp", &tol);
    }
}

static void set_initial_guess(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_in *in,
    ocp_nlp_out *out)
{
    double x_init[NX] = {-1.2, 0, 0};
    double u_init[NU] = {0, 0};
    for (int k = 0; k < N; ++k) {
        ocp_nlp_out_set(config, dims, out, in, k, "x", x_init);
        ocp_nlp_out_set(config, dims, out, in, k, "u", u_init);
    }
    ocp_nlp_out_set(config, dims, out, in, N, "x", x_init);
}

static void set_initial_state_constraint(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_in *in,
    ocp_nlp_out *out)
{
    double lbx0[NBX0] = {-1.2, 0, 0};
    double ubx0[NBX0] = {-1.2, 0, 0};
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "ubx", ubx0);
}

static void copy_field(
    ocp_nlp_config *src_config,
    ocp_nlp_dims *src_dims,
    ocp_nlp_in *src_in,
    ocp_nlp_out *src_out,
    ocp_nlp_config *dst_config,
    ocp_nlp_dims *dst_dims,
    ocp_nlp_in *dst_in,
    ocp_nlp_out *dst_out,
    int stage,
    const char *field)
{
    double buffer[128] = {0.0};
    int dim = ocp_nlp_dims_get_from_attr(src_config, src_dims, src_out, stage, field);
    if (dim <= 0) {
        return;
    }
    if (dim > (int)(sizeof(buffer) / sizeof(buffer[0]))) {
        printf("copy_field_error=%s_dim_%d_too_large\n", field, dim);
        exit(2);
    }
    ocp_nlp_out_get(src_config, src_dims, src_out, stage, field, buffer);
    ocp_nlp_out_set(dst_config, dst_dims, dst_out, dst_in, stage, field, buffer);
}

static void copy_forward_iterate_to_sensitivity(
    ocp_nlp_config *f_config,
    ocp_nlp_dims *f_dims,
    ocp_nlp_in *f_in,
    ocp_nlp_out *f_out,
    ocp_nlp_config *s_config,
    ocp_nlp_dims *s_dims,
    ocp_nlp_in *s_in,
    ocp_nlp_out *s_out)
{
    for (int k = 0; k <= N; ++k) {
        copy_field(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out, k, "x");
        copy_field(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out, k, "lam");
        copy_field(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out, k, "sl");
        copy_field(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out, k, "su");
        if (k < N) {
            copy_field(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out, k, "u");
            copy_field(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out, k, "pi");
        }
    }
}

static void seed_adjoint_from_controls(
    ocp_nlp_config *config,
    ocp_nlp_dims *dims,
    ocp_nlp_in *in,
    ocp_nlp_out *solution_out,
    ocp_nlp_out *sens_out)
{
    ocp_nlp_out_set_values_to_zero(config, dims, sens_out);
    for (int k = 0; k < N; ++k) {
        double seed_u[NU] = {0.0, 0.0};
        ocp_nlp_out_get(config, dims, solution_out, k, "u", seed_u);
        ocp_nlp_out_set(config, dims, sens_out, in, k, "u", seed_u);
    }
}

int main(void)
{
    const double acados_tol = env_double("ACADOS_TOL", -1.0);
    double p_global[NP_GLOBAL] = {
        10, 0.20000000000000001, 0.01, 0.01, 30,
        0.29999999999999999, 0.20000000000000001,
        -1.2, 0, 0, 1.2, 0, 0
    };

    circle_forward_solver_capsule *forward = circle_forward_acados_create_capsule();
    circle_sensitivity_solver_capsule *sensitivity = circle_sensitivity_acados_create_capsule();
    int status = circle_forward_acados_create_with_discretization(forward, N, NULL);
    if (status) {
        printf("forward_create_status=%d\n", status);
        return status;
    }
    status = circle_sensitivity_acados_create_with_discretization(sensitivity, N, NULL);
    if (status) {
        printf("sensitivity_create_status=%d\n", status);
        return status;
    }

    ocp_nlp_config *f_config = circle_forward_acados_get_nlp_config(forward);
    ocp_nlp_dims *f_dims = circle_forward_acados_get_nlp_dims(forward);
    ocp_nlp_in *f_in = circle_forward_acados_get_nlp_in(forward);
    ocp_nlp_out *f_out = circle_forward_acados_get_nlp_out(forward);
    ocp_nlp_solver *f_solver = circle_forward_acados_get_nlp_solver(forward);
    void *f_opts = circle_forward_acados_get_nlp_opts(forward);

    ocp_nlp_config *s_config = circle_sensitivity_acados_get_nlp_config(sensitivity);
    ocp_nlp_dims *s_dims = circle_sensitivity_acados_get_nlp_dims(sensitivity);
    ocp_nlp_in *s_in = circle_sensitivity_acados_get_nlp_in(sensitivity);
    ocp_nlp_out *s_out = circle_sensitivity_acados_get_nlp_out(sensitivity);
    ocp_nlp_out *s_sens_out = circle_sensitivity_acados_get_sens_out(sensitivity);
    ocp_nlp_solver *s_solver = circle_sensitivity_acados_get_nlp_solver(sensitivity);
    void *s_opts = circle_sensitivity_acados_get_nlp_opts(sensitivity);

    override_tolerances(f_config, f_opts, acados_tol);
    override_tolerances(s_config, s_opts, acados_tol);

    status = circle_forward_acados_set_p_global_and_precompute_dependencies(forward, p_global, NP_GLOBAL);
    if (status) {
        printf("forward_p_global_status=%d\n", status);
        return status;
    }
    status = circle_sensitivity_acados_set_p_global_and_precompute_dependencies(sensitivity, p_global, NP_GLOBAL);
    if (status) {
        printf("sensitivity_p_global_status=%d\n", status);
        return status;
    }

    set_initial_state_constraint(f_config, f_dims, f_in, f_out);
    set_initial_state_constraint(s_config, s_dims, s_in, s_out);
    set_initial_guess(f_config, f_dims, f_in, f_out);
    set_initial_guess(s_config, s_dims, s_in, s_out);

    const double forward_t0 = wall_now_sec();
    status = circle_forward_acados_solve(forward);
    const double forward_time_sec = wall_now_sec() - forward_t0;

    int backward_status = -1;
    double backward_time_sec = 0.0;
    double grad_p[NP_GLOBAL] = {0.0};
    if (status == 0) {
        const double backward_t0 = wall_now_sec();
        copy_forward_iterate_to_sensitivity(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out);
        backward_status = circle_sensitivity_acados_setup_qp_matrices_and_factorize(sensitivity);
        if (backward_status == 0) {
            ocp_nlp_eval_params_jac(s_solver, s_in, s_out);
            seed_adjoint_from_controls(s_config, s_dims, s_in, s_out, s_sens_out);
            ocp_nlp_eval_solution_sens_adj_p(s_solver, s_in, s_sens_out, "p_global", 0, grad_p);
        }
        backward_time_sec = wall_now_sec() - backward_t0;
    }

    double acados_time_sec = 0.0;
    double kkt_norm_inf = 0.0;
    int sqp_iter = -1;
    double u0_out[NU] = {0.0, 0.0};
    ocp_nlp_get(f_solver, "time_tot", &acados_time_sec);
    ocp_nlp_get(f_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_out_get(f_config, f_dims, f_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_out_get(f_config, f_dims, f_out, 0, "u", u0_out);

    double grad_norm = 0.0;
    for (int i = 0; i < NP_GLOBAL; ++i) {
        grad_norm += grad_p[i] * grad_p[i];
    }
    grad_norm = sqrt(grad_norm);

    printf("solve_status=%d\n", status);
    printf("backward_status=%d\n", backward_status);
    printf("acados_tol=%.17g\n", acados_tol);
    printf("sqp_iter=%d\n", sqp_iter);
    printf("time_tot_sec=%.17g\n", acados_time_sec);
    printf("forward_time_sec=%.17g\n", forward_time_sec);
    printf("backward_time_sec=%.17g\n", backward_time_sec);
    printf("kkt_norm_inf=%.17g\n", kkt_norm_inf);
    printf("u0=%.17g,%.17g\n", u0_out[0], u0_out[1]);
    printf("grad_p_global_norm=%.17g\n", grad_norm);
    printf("grad_p_global=");
    for (int i = 0; i < NP_GLOBAL; ++i) {
        printf("%s%.17g", (i == 0) ? "" : ",", grad_p[i]);
    }
    printf("\n");

    int free_status = circle_forward_acados_free(forward);
    int capsule_status = circle_forward_acados_free_capsule(forward);
    free_status |= circle_sensitivity_acados_free(sensitivity);
    capsule_status |= circle_sensitivity_acados_free_capsule(sensitivity);
    if (free_status || capsule_status) {
        printf("free_status=%d capsule_status=%d\n", free_status, capsule_status);
    }
    return status || backward_status;
}


