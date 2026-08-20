#include <stdio.h>
#include <stdlib.h>

#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_rect_margin_forward.h"

#define NX RECT_MARGIN_FORWARD_NX
#define NU RECT_MARGIN_FORWARD_NU
#define N  RECT_MARGIN_FORWARD_N
#define NBX0 RECT_MARGIN_FORWARD_NBX0
#define NP_GLOBAL RECT_MARGIN_FORWARD_NP_GLOBAL

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

int main(void)
{
    rect_margin_forward_solver_capsule *capsule = rect_margin_forward_acados_create_capsule();
    int status = rect_margin_forward_acados_create_with_discretization(capsule, N, NULL);
    if (status) {
        printf("create_status=%d\n", status);
        return status;
    }

    ocp_nlp_config *config = rect_margin_forward_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = rect_margin_forward_acados_get_nlp_dims(capsule);
    ocp_nlp_in *in = rect_margin_forward_acados_get_nlp_in(capsule);
    ocp_nlp_out *out = rect_margin_forward_acados_get_nlp_out(capsule);
    ocp_nlp_solver *solver = rect_margin_forward_acados_get_nlp_solver(capsule);

    double p_global[NP_GLOBAL] = {5, 0.20000000000000001, 0.01, 0.01, 20, 0.14999999999999999, 0.14999999999999999, 0.16, 0.22, -1.2, 0, 0, 1.2, 0, 0};
    status = rect_margin_forward_acados_set_p_global_and_precompute_dependencies(capsule, p_global, NP_GLOBAL);
    if (status) {
        printf("p_global_status=%d\n", status);
        return status;
    }

    double lbx0[NBX0] = {-1.2, 0, 0};
    double ubx0[NBX0] = {-1.2, 0, 0};
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "ubx", ubx0);

    set_initial_guess(config, dims, in, out);
    status = rect_margin_forward_acados_solve(capsule);

    double elapsed_time = 0.0;
    double kkt_norm_inf = 0.0;
    int sqp_iter = -1;
    double u0_out[NU] = {0.0};
    ocp_nlp_get(solver, "time_tot", &elapsed_time);
    ocp_nlp_get(solver, "sqp_iter", &sqp_iter);
    ocp_nlp_out_get(config, dims, out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_out_get(config, dims, out, 0, "u", u0_out);

    printf("solve_status=%d\n", status);
    printf("sqp_iter=%d\n", sqp_iter);
    printf("time_tot_sec=%.17g\n", elapsed_time);
    printf("kkt_norm_inf=%.17g\n", kkt_norm_inf);
    printf("u0=%.17g,%.17g\n", u0_out[0], u0_out[1]);

    int free_status = rect_margin_forward_acados_free(capsule);
    int capsule_status = rect_margin_forward_acados_free_capsule(capsule);
    if (free_status || capsule_status) {
        printf("free_status=%d capsule_status=%d\n", free_status, capsule_status);
    }
    return status;
}
