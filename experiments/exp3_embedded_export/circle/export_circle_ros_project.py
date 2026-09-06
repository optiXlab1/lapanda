"""Export the circle obstacle MPC as a standalone C/ROS-ready project."""

from __future__ import annotations

import argparse
import importlib.util
import json
import stat
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import config as exp3_config
from lapanda import export_c_project

_circle_spec = importlib.util.spec_from_file_location(
    "circle_lapanda_impl", Path(__file__).resolve().parent / "run_lapanda.py"
)
if _circle_spec is None or _circle_spec.loader is None:
    raise ImportError("cannot load circle/run_lapanda.py")
_circle_impl = importlib.util.module_from_spec(_circle_spec)
sys.modules[_circle_spec.name] = _circle_impl
_circle_spec.loader.exec_module(_circle_impl)

build_circle_case = _circle_impl.build_circle_case


MAIN_C = r'''#include "static_casadi_oracle.h"
#include "lapanda_generated_config.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static void fill_problem_data(double* theta, double* variable)
{
    const double theta_value[LAPANDA_NTHETA] = {
        10.0, 0.2, 1e-2, 1e-2, 30.0, 0.30, 0.20
    };
    const double variable_value[LAPANDA_NVAR] = {
        -1.2, 0.0, 0.0, 1.2, 0.0, 0.0
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
    solver_params.tolerance = 1e-1;
    solver_params.buffer_size = 10;
    solver_params.max_stable_iter = 80;
    solver_params.verbose = 0;

    backward_params.enable = 1;
    backward_params.tolerance = 1e-3;
    backward_params.max_iterations = 200;

    lapanda_static_init_alm_problem(
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
'''


def patch_cmake(project_dir: Path) -> None:
    cmake_path = project_dir / "CMakeLists.txt"
    text = cmake_path.read_text(encoding="utf-8")
    if "target_link_libraries(lapanda_embedded PUBLIC m)" not in text:
        text = text.replace(
            "target_include_directories(lapanda_embedded PUBLIC",
            "if(UNIX)\n"
            "    target_link_libraries(lapanda_embedded PUBLIC m)\n"
            "endif()\n\n"
            "target_include_directories(lapanda_embedded PUBLIC",
        )
    addition = """

add_executable(circle_mpc_benchmark main_circle_benchmark.c)
target_link_libraries(circle_mpc_benchmark PRIVATE lapanda_embedded)
"""
    if "circle_mpc_benchmark" not in text:
        text = text.rstrip() + addition
    cmake_path.write_text(text, encoding="utf-8")


ROS1_CPP = r'''#include <algorithm>
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

class CircleMpcNode {
public:
    CircleMpcNode()
        : nh_()
        , private_nh_("~")
    {
        private_nh_.param("period_sec", period_sec_, 0.05);
        private_nh_.param("inner_max_iterations", inner_max_iterations_, 2000);
        private_nh_.param("inner_tolerance", inner_tolerance_, 1e-1);
        private_nh_.param("alm_max_iterations", alm_max_iterations_, 100);
        private_nh_.param("alm_tolerance", alm_tolerance_, 1e-4);
        private_nh_.param("initial_penalty", initial_penalty_, 10000.0);
        private_nh_.param("penalty_update_factor", penalty_update_factor_, 10.0);
        private_nh_.param("compute_backward", compute_backward_, false);

        theta_ = {10.0, 0.2, 1e-2, 1e-2, 30.0, 0.30, 0.20};
        variable_ = {-1.2, 0.0, 0.0, 1.2, 0.0, 0.0};
        solution_.fill(0.0);
        multipliers_.fill(0.0);

        for (std::size_t i = 0; i < kNCon; ++i) {
            constraint_lower_[i] = -1.0e20;
            constraint_upper_[i] = 0.0;
        }

        state_sub_ = nh_.subscribe("state", 10, &CircleMpcNode::stateCallback, this);
        target_sub_ = nh_.subscribe("target", 10, &CircleMpcNode::targetCallback, this);
        theta_sub_ = nh_.subscribe("theta", 10, &CircleMpcNode::thetaCallback, this);
        obstacle_sub_ = nh_.subscribe("circle_obstacle", 10, &CircleMpcNode::obstacleCallback, this);

        control_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("control", 10);
        info_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("solver_info", 10);
        timer_ = nh_.createTimer(ros::Duration(period_sec_), &CircleMpcNode::timerCallback, this);
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

    void obstacleCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
    {
        // [safe_radius, obstacle_y]
        if (msg->data.size() >= 2) {
            theta_[5] = msg->data[0];
            theta_[6] = msg->data[1];
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
            ROS_WARN("lapanda circle solve failed with status %d", status);
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
    ros::Subscriber obstacle_sub_;
    ros::Publisher control_pub_;
    ros::Publisher info_pub_;
    ros::Timer timer_;

    double period_sec_;
    int inner_max_iterations_;
    int alm_max_iterations_;
    double inner_tolerance_;
    double alm_tolerance_;
    double initial_penalty_;
    double penalty_update_factor_;
    bool compute_backward_;

    std::array<double, kN> solution_{};
    std::array<double, kNTheta> theta_{};
    std::array<double, kNVar> variable_{};
    std::array<double, kNCon> multipliers_{};
    std::array<double, kNCon> constraint_lower_{};
    std::array<double, kNCon> constraint_upper_{};
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "circle_mpc_node");
    CircleMpcNode node;
    ros::spin();
    return 0;
}
'''


def write_ros1_package(project_dir: Path) -> None:
    package_dir = project_dir / "ros1_circle_mpc"
    src_dir = package_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "circle_mpc_node.cpp").write_text(ROS1_CPP, encoding="utf-8")
    (package_dir / "package.xml").write_text(
        """<?xml version=\"1.0\"?>
<package format=\"2\">
  <name>ros1_circle_mpc</name>
  <version>0.1.0</version>
  <description>ROS1 wrapper for the exported lapanda circle MPC.</description>
  <maintainer email=\"user@example.com\">lapanda user</maintainer>
  <license>MIT</license>
  <buildtool_depend>catkin</buildtool_depend>
  <build_depend>roscpp</build_depend>
  <build_depend>std_msgs</build_depend>
  <exec_depend>roscpp</exec_depend>
  <exec_depend>std_msgs</exec_depend>
</package>
""",
        encoding="utf-8",
    )
    (package_dir / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.10)
project(ros1_circle_mpc)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

catkin_package()

set(lapanda_EXPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(lapanda_ROOT ${lapanda_EXPORT_DIR}/lapanda)

add_library(lapanda_circle STATIC
    ${lapanda_EXPORT_DIR}/generated/circle_mpc_oracle.c
    ${lapanda_ROOT}/adapters/casadi_static/static_casadi_oracle.c
    ${lapanda_ROOT}/panda/function_evaluator.c
    ${lapanda_ROOT}/panda/matrix_operations.c
    ${lapanda_ROOT}/panda/lbfgs.c
    ${lapanda_ROOT}/panda/panda_eval.c
    ${lapanda_ROOT}/panda/panda_forward.c
    ${lapanda_ROOT}/panda/panda_linear_solver.c
    ${lapanda_ROOT}/panda/panda_backward.c
    ${lapanda_ROOT}/panda/optimizer.c
    ${lapanda_ROOT}/alm/alm.c
)

target_include_directories(lapanda_circle PUBLIC
    ${lapanda_EXPORT_DIR}/generated
    ${lapanda_ROOT}/adapters/casadi_static
    ${lapanda_ROOT}/include
    ${lapanda_ROOT}/panda
    ${lapanda_ROOT}/alm
    ${lapanda_ROOT}/globals
)

if(UNIX)
    target_link_libraries(lapanda_circle PUBLIC m)
endif()

add_executable(circle_mpc_node src/circle_mpc_node.cpp)
target_include_directories(circle_mpc_node PRIVATE ${catkin_INCLUDE_DIRS})
target_link_libraries(circle_mpc_node
    lapanda_circle
    ${catkin_LIBRARIES}
)
add_dependencies(circle_mpc_node ${catkin_EXPORTED_TARGETS})
""",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(
        """# ROS1 Circle MPC

Place `exports/circle_ros_mpc` in a catkin workspace, then build:

```bash
catkin_make --pkg ros1_circle_mpc
source devel/setup.bash
rosrun ros1_circle_mpc circle_mpc_node _compute_backward:=true
```

Topics:
- subscribe `state`: `[x, y, heading]`
- subscribe `target`: `[x, y, heading]`
- subscribe `circle_obstacle`: `[safe_radius, obstacle_y]`
- subscribe `theta`: full 7-vector override
- publish `control`: first `[speed, steer]`
- publish `solver_info`: `[status, outer_iter, residual, penalty, forward_time, backward_time, backward_iter]`
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(REPO_ROOT / "exports" / "circle_ros_mpc"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--horizon", type=int, default=exp3_config.CIRCLE_HORIZON)
    parser.add_argument("--speed-limit", type=float, default=exp3_config.CIRCLE_SPEED_LIMIT)
    return parser.parse_args()


def write_run_scripts(project_dir: Path) -> None:
    standalone = project_dir / "run_standalone.sh"
    standalone.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$(nproc)"
"$ROOT/build/circle_mpc_benchmark"
""",
        encoding="utf-8",
    )
    ros_runner = project_dir / "run_ros1_node.sh"
    ros_runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
: "${ROS_DISTRO:=noetic}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="$ROOT/_catkin_ws"
mkdir -p "$WS/src"
ln -sfn "$ROOT/ros1_circle_mpc" "$WS/src/ros1_circle_mpc"
catkin_make -C "$WS"
source "$WS/devel/setup.bash"
extra_args=()
if [ -n "${ALM_TOL:-}" ]; then
  extra_args+=("_alm_tolerance:=${ALM_TOL}")
fi
rosrun ros1_circle_mpc circle_mpc_node "${extra_args[@]}"
""",
        encoding="utf-8",
    )
    for script in (standalone, ros_runner):
        content = script.read_text(encoding="utf-8")
        with script.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> None:
    args = parse_args()
    case = build_circle_case(args.horizon, args.speed_limit)
    case.theta[:] = exp3_config.CIRCLE_THETA
    outdir = export_c_project(
        case.problem,
        args.outdir,
        name="circle_mpc_oracle",
        force=args.force,
    )
    (outdir / "main_circle_benchmark.c").write_text(MAIN_C, encoding="utf-8")
    patch_cmake(outdir)
    write_ros1_package(outdir)
    write_run_scripts(outdir)
    (outdir / "export_manifest.json").write_text(
        json.dumps(
            {
                "generator": (
                    "experiments/exp3_embedded_export/circle/"
                    "export_circle_ros_project.py"
                ),
                "files": sorted(
                    str(path.relative_to(outdir)).replace("\\", "/")
                    for path in outdir.rglob("*")
                    if path.is_file() and path.name != "export_manifest.json"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(outdir)


if __name__ == "__main__":
    main()
