#include <cmath>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_circle_forward.h"
#include "acados_solver_circle_sensitivity.h"
}

#define NX CIRCLE_FORWARD_NX
#define NU CIRCLE_FORWARD_NU
#define N  CIRCLE_FORWARD_N
#define NBX0 CIRCLE_FORWARD_NBX0
#define NP_GLOBAL CIRCLE_FORWARD_NP_GLOBAL

static void override_tolerances(ocp_nlp_config *config, void *opts, double tol)
{
    if (tol > 0.0) {
        ocp_nlp_solver_opts_set(config, opts, "tol_stat", &tol);
        ocp_nlp_solver_opts_set(config, opts, "tol_eq", &tol);
        ocp_nlp_solver_opts_set(config, opts, "tol_ineq", &tol);
        ocp_nlp_solver_opts_set(config, opts, "tol_comp", &tol);
    }
}

static void set_initial_guess(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in, ocp_nlp_out *out)
{
    double x_init[NX] = {-1.2, 0, 0};
    double u_init[NU] = {0, 0};
    for (int k = 0; k < N; ++k) {
        ocp_nlp_out_set(config, dims, out, in, k, "x", x_init);
        ocp_nlp_out_set(config, dims, out, in, k, "u", u_init);
    }
    ocp_nlp_out_set(config, dims, out, in, N, "x", x_init);
}

static void set_initial_state_constraint(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in, ocp_nlp_out *out)
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
    const int dim = ocp_nlp_dims_get_from_attr(src_config, src_dims, src_out, stage, field);
    if (dim <= 0) {
        return;
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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "circle_acados_node");
    ros::NodeHandle nh("~");
    ros::Publisher pub = nh.advertise<std_msgs::Float64MultiArray>("solve_info", 1, true);

    double acados_tol = -1.0;
    nh.param("acados_tol", acados_tol, -1.0);
    double p_global[NP_GLOBAL] = {10, 0.2, 0.01, 0.01, 30, 0.3, 0.2, -1.2, 0, 0, 1.2, 0, 0};

    circle_forward_solver_capsule *forward = circle_forward_acados_create_capsule();
    circle_sensitivity_solver_capsule *sensitivity = circle_sensitivity_acados_create_capsule();
    int status = circle_forward_acados_create_with_discretization(forward, N, nullptr);
    if (status) {
        ROS_ERROR_STREAM("forward acados create failed: " << status);
        return status;
    }
    status = circle_sensitivity_acados_create_with_discretization(sensitivity, N, nullptr);
    if (status) {
        ROS_ERROR_STREAM("sensitivity acados create failed: " << status);
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
    status |= circle_sensitivity_acados_set_p_global_and_precompute_dependencies(sensitivity, p_global, NP_GLOBAL);
    if (status) {
        ROS_ERROR_STREAM("set_p_global failed: " << status);
        return status;
    }

    set_initial_state_constraint(f_config, f_dims, f_in, f_out);
    set_initial_state_constraint(s_config, s_dims, s_in, s_out);
    set_initial_guess(f_config, f_dims, f_in, f_out);
    set_initial_guess(s_config, s_dims, s_in, s_out);

    const ros::WallTime forward_t0 = ros::WallTime::now();
    status = circle_forward_acados_solve(forward);
    const double forward_time_sec = (ros::WallTime::now() - forward_t0).toSec();

    int backward_status = -1;
    double backward_time_sec = 0.0;
    double grad_p[NP_GLOBAL] = {0.0};
    if (status == 0) {
        const ros::WallTime backward_t0 = ros::WallTime::now();
        copy_forward_iterate_to_sensitivity(f_config, f_dims, f_in, f_out, s_config, s_dims, s_in, s_out);
        backward_status = circle_sensitivity_acados_setup_qp_matrices_and_factorize(sensitivity);
        if (backward_status == 0) {
            ocp_nlp_eval_params_jac(s_solver, s_in, s_out);
            seed_adjoint_from_controls(s_config, s_dims, s_in, s_out, s_sens_out);
            ocp_nlp_eval_solution_sens_adj_p(s_solver, s_in, s_sens_out, "p_global", 0, grad_p);
        }
        backward_time_sec = (ros::WallTime::now() - backward_t0).toSec();
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
    grad_norm = std::sqrt(grad_norm);

    std_msgs::Float64MultiArray msg;
    msg.data = {
        static_cast<double>(status),
        static_cast<double>(backward_status),
        static_cast<double>(sqp_iter),
        acados_time_sec,
        forward_time_sec,
        backward_time_sec,
        kkt_norm_inf,
        u0_out[0],
        u0_out[1],
        acados_tol,
        grad_norm,
    };
    pub.publish(msg);
    ROS_INFO_STREAM("circle: status=" << status
                    << " backward_status=" << backward_status
                    << " sqp_iter=" << sqp_iter
                    << " forward_time_sec=" << forward_time_sec
                    << " backward_time_sec=" << backward_time_sec
                    << " grad_norm=" << grad_norm);

    circle_forward_acados_free(forward);
    circle_forward_acados_free_capsule(forward);
    circle_sensitivity_acados_free(sensitivity);
    circle_sensitivity_acados_free_capsule(sensitivity);
    ros::spinOnce();
    return status || backward_status;
}


