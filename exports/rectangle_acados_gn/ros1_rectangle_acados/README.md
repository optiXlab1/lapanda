# rectangle_acados_gn_ros1

ROS1 wrapper for the acados C interface rectangle solver.

The node publishes `/~solve_info` as `std_msgs/Float64MultiArray`:

1. solve status
2. SQP iterations
3. acados reported solve time [s]
4. ROS wall-clock solve time [s]
5. KKT infinity norm
6. first control speed
7. first control steering
