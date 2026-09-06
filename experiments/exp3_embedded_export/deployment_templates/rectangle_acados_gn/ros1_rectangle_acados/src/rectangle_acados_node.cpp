#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>

extern "C" {
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_rect_margin_forward.h"
}

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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rectangle_acados_node");
    ros::NodeHandle nh("~");
    ros::Publisher pub = nh.advertise<std_msgs::Float64MultiArray>("solve_info", 1, true);

    rect_margin_forward_solver_capsule *capsule = rect_margin_forward_acados_create_capsule();
    int status = rect_margin_forward_acados_create_with_discretization(capsule, N, nullptr);
    if (status) {
        ROS_ERROR_STREAM("acados create failed: " << status);
        return status;
    }

    ocp_nlp_config *config = rect_margin_forward_acados_get_nlp_config(capsule);
    ocp_nlp_dims *dims = rect_margin_forward_acados_get_nlp_dims(capsule);
    ocp_nlp_in *in = rect_margin_forward_acados_get_nlp_in(capsule);
    ocp_nlp_out *out = rect_margin_forward_acados_get_nlp_out(capsule);
    ocp_nlp_solver *solver = rect_margin_forward_acados_get_nlp_solver(capsule);

    double p_global[NP_GLOBAL] = {5, 0.20000000000000001, 0.01, 0.01, 20, 0.12, 0.12, 0.01, 0.22, -1.2, 0, 0, 1.2, 0, 0};
    status = rect_margin_forward_acados_set_p_global_and_precompute_dependencies(capsule, p_global, NP_GLOBAL);
    if (status) {
        ROS_ERROR_STREAM("set_p_global failed: " << status);
        return status;
    }

    double lbx0[NBX0] = {-1.2, 0, 0};
    double ubx0[NBX0] = {-1.2, 0, 0};
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(config, dims, in, out, 0, "ubx", ubx0);

    set_initial_guess(config, dims, in, out);
    const ros::WallTime t0 = ros::WallTime::now();
    status = rect_margin_forward_acados_solve(capsule);
    const double wall_time_sec = (ros::WallTime::now() - t0).toSec();

    double acados_time_sec = 0.0;
    double kkt_norm_inf = 0.0;
    int sqp_iter = -1;
    double u0_out[NU] = {0.0};
    ocp_nlp_get(solver, "time_tot", &acados_time_sec);
    ocp_nlp_get(solver, "sqp_iter", &sqp_iter);
    ocp_nlp_out_get(config, dims, out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_out_get(config, dims, out, 0, "u", u0_out);

    std_msgs::Float64MultiArray msg;
    msg.data = {
        static_cast<double>(status),
        static_cast<double>(sqp_iter),
        acados_time_sec,
        wall_time_sec,
        kkt_norm_inf,
        u0_out[0],
        u0_out[1],
    };
    pub.publish(msg);
    ROS_INFO_STREAM("rectangle: status=" << status
                    << " sqp_iter=" << sqp_iter
                    << " acados_time_sec=" << acados_time_sec
                    << " wall_time_sec=" << wall_time_sec
                    << " kkt=" << kkt_norm_inf
                    << " u0=[" << u0_out[0] << ", " << u0_out[1] << "]");

    rect_margin_forward_acados_free(capsule);
    rect_margin_forward_acados_free_capsule(capsule);
    ros::spinOnce();
    return status;
}

