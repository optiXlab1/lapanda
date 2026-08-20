# ROS1 Circle MPC

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
