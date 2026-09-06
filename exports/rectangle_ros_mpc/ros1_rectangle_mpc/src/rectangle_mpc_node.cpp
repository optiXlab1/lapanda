#include <algorithm>
#include <array>
#include <cstddef>

#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"

extern "C" {
#include "static_casadi_oracle.h"
#include "lapanda_generated_config.h"
}

namespace {
constexpr std::size_t kN = LAPANDA_N;
constexpr std::size_t kNTheta = LAPANDA_NTHETA;
constexpr std::size_t kNVar = (LAPANDA_NVAR > 0 ? LAPANDA_NVAR : 1);
constexpr std::size_t kNCon = LAPANDA_NCON;

template <std::size_t N>
void copy_prefix(const std_msgs::Float64MultiArray::ConstPtr& msg, std::array<double, N>& dst)
{
    const std::size_t count = std::min<std::size_t>(N, msg->data.size());
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] = msg->data[i];
    }
}
}  // namespace

class RectangleMpcNode {
public:
    RectangleMpcNode()
        : nh_()
        , private_nh_("~")
    {
        private_nh_.param("period_sec", period_sec_, 0.05);
        private_nh_.param("compute_backward", compute_backward_, false);
        private_nh_.param("inner_max_iterations", inner_max_iterations_, 2000);
        private_nh_.param("inner_tolerance", inner_tolerance_, 1e-3);
        private_nh_.param("alm_max_iterations", alm_max_iterations_, 100);
        private_nh_.param("alm_tolerance", alm_tolerance_, 1e-5);
        private_nh_.param("initial_penalty", initial_penalty_, 10000.0);
        private_nh_.param("penalty_update_factor", penalty_update_factor_, 10.0);

        theta_ = {5.0, 0.2, 1e-2, 1e-2, 20.0, 0.12, 0.12, 0.01, 0.22};
        variable_.fill(0.0);
        variable_[0] = -1.2;
        variable_[3] = 1.2;
        solution_.fill(0.0);
        multipliers_.fill(0.0);
        for (std::size_t i = 0; i < kNCon; ++i) {
            constraint_lower_[i] = -1.0e20;
            constraint_upper_[i] = 0.0;
        }

        state_sub_ = nh_.subscribe("state", 10, &RectangleMpcNode::stateCallback, this);
        target_sub_ = nh_.subscribe("target", 10, &RectangleMpcNode::targetCallback, this);
        theta_sub_ = nh_.subscribe("theta", 10, &RectangleMpcNode::thetaCallback, this);
        teacher_sub_ = nh_.subscribe("teacher_control", 10, &RectangleMpcNode::teacherCallback, this);
        margins_sub_ = nh_.subscribe("rectangle_margins", 10, &RectangleMpcNode::marginsCallback, this);

        control_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("control", 10);
        info_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("solver_info", 10);
        timer_ = nh_.createTimer(ros::Duration(period_sec_), &RectangleMpcNode::timerCallback, this);
    }

private:
    void stateCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        if (msg->data.size() >= 3) {
            variable_[0] = msg->data[0];
            variable_[1] = msg->data[1];
            variable_[2] = msg->data[2];
        }
    }

    void targetCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        if (msg->data.size() >= 3) {
            variable_[3] = msg->data[0];
            variable_[4] = msg->data[1];
            variable_[5] = msg->data[2];
        }
    }

    void thetaCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        copy_prefix(msg, theta_);
    }

    void teacherCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        const std::size_t count = std::min<std::size_t>(kN, msg->data.size());
        for (std::size_t i = 0; i < count; ++i) {
            variable_[6 + i] = msg->data[i];
        }
    }

    void marginsCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        if (msg->data.size() >= 4) {
            theta_[5] = msg->data[0];
            theta_[6] = msg->data[1];
            theta_[7] = msg->data[2];
            theta_[8] = msg->data[3];
        }
    }

    void timerCallback(const ros::TimerEvent&)
    {
        solver_parameters inner{};
        inner.max_iterations = static_cast<unsigned int>(inner_max_iterations_);
        inner.tolerance = inner_tolerance_;
        inner.buffer_size = 10;
        inner.max_stable_iter = 80;
        inner.verbose = 0;

        backward_parameters backward{};
        backward.enable = compute_backward_ ? 1 : 0;
        backward.tolerance = 1e-3;
        backward.max_iterations = 200;

        alm_problem problem{};
        lapanda_static_init_alm_problem(
            &problem,
            constraint_lower_.data(),
            constraint_upper_.data(),
            &inner,
            &backward);

        alm_parameters params{};
        params.max_iterations = static_cast<unsigned int>(alm_max_iterations_);
        params.tolerance = alm_tolerance_;
        params.initial_penalty = initial_penalty_;
        params.penalty_update_factor = penalty_update_factor_;
        params.max_penalty = 0.0;
        params.sufficient_decrease_factor = 0.25;
        params.verbose = 0;
        params.warm_start_inner = 1;

        alm_info info{};
        optimizer_solve_info backward_info{};
        std::array<double, kNTheta> grad{};
        int status = 0;
        if (compute_backward_) {
            status = alm_solve_with_backward_and_penalty0(
                &problem,
                &params,
                solution_.data(),
                multipliers_.data(),
                nullptr,
                theta_.data(),
                variable_.data(),
                &info,
                grad.data(),
                &backward_info);
        } else {
            status = alm_solve(
                &problem,
                &params,
                solution_.data(),
                multipliers_.data(),
                theta_.data(),
                variable_.data(),
                &info);
        }

        if (status < 0) {
            ROS_WARN("lapanda rectangle solve failed with status %d", status);
            return;
        }

        std_msgs::Float64MultiArray control_msg;
        control_msg.data.push_back(solution_[0]);
        control_msg.data.push_back(solution_[1]);
        control_pub_.publish(control_msg);

        std_msgs::Float64MultiArray info_msg;
        info_msg.data.push_back(static_cast<double>(status));
        info_msg.data.push_back(static_cast<double>(info.iterations));
        info_msg.data.push_back(info.final_residual);
        info_msg.data.push_back(info.penalty);
        info_msg.data.push_back(info.forward_time_sec);
        info_msg.data.push_back(info.backward_time_sec);
        info_msg.data.push_back(static_cast<double>(backward_info.iterations));
        info_pub_.publish(info_msg);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Subscriber state_sub_;
    ros::Subscriber target_sub_;
    ros::Subscriber theta_sub_;
    ros::Subscriber teacher_sub_;
    ros::Subscriber margins_sub_;
    ros::Publisher control_pub_;
    ros::Publisher info_pub_;
    ros::Timer timer_;

    double period_sec_;
    bool compute_backward_;
    int inner_max_iterations_;
    int alm_max_iterations_;
    double inner_tolerance_;
    double alm_tolerance_;
    double initial_penalty_;
    double penalty_update_factor_;

    std::array<double, kN> solution_{};
    std::array<double, kNTheta> theta_{};
    std::array<double, kNVar> variable_{};
    std::array<double, kNCon> multipliers_{};
    std::array<double, kNCon> constraint_lower_{};
    std::array<double, kNCon> constraint_upper_{};
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rectangle_mpc_node");
    RectangleMpcNode node;
    ros::spin();
    return 0;
}
