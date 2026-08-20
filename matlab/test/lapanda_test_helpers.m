function h = lapanda_test_helpers()
%lapanda_TEST_HELPERS Shared model builders for MATLAB obstacle tests.

h.setup_paths = @setup_paths;
h.build_circle_problem = @build_circle_problem;
h.build_rectangle_problem = @build_rectangle_problem;
h.build_quadratic_problem = @build_quadratic_problem;
end

function repo_root = setup_paths()
this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(repo_root);
addpath(fullfile(repo_root, 'matlab'));
casadi_path = getenv('CASADI_MATLAB_PATH');
if isempty(casadi_path)
    bundled = fullfile(repo_root, 'matlab', 'casadi-3.7.2-windows64-matlab2018b');
    if exist(bundled, 'dir')
        casadi_path = bundled;
    end
end
if ~isempty(casadi_path)
    addpath(genpath(casadi_path));
end
end

function [problem, meta] = build_circle_problem()
import casadi.*
horizon = 30;
dt = 0.12;
nu = 2;
n = horizon * nu;
u = SX.sym('u', n, 1);
theta = SX.sym('theta', 6, 1);
variable = SX.sym('variable', 6, 1);

position_weight = theta(1);
heading_weight = theta(2);
speed_weight = theta(3);
steer_weight = theta(4);
terminal_weight = theta(5);
safe_radius = theta(6);
px = variable(1);
py = variable(2);
heading = variable(3);
target_x = variable(4);
target_y = variable(5);
target_heading = variable(6);
obstacle = [0; 0];
wheelbase = 0.45;

cost = 0;
constraints = SX.zeros(horizon, 1);
for k = 1:horizon
    uk = u(2*k-1:2*k);
    [px, py, heading] = bicycle_step(px, py, heading, uk, dt, wheelbase);
    dx = px - target_x;
    dy = py - target_y;
    dheading = angle_error(heading, target_heading);
    cost = cost + position_weight * (dx^2 + dy^2) ...
        + heading_weight * dheading^2 ...
        + speed_weight * uk(1)^2 ...
        + steer_weight * uk(2)^2;
    constraints(k) = safe_radius^2 - ((px - obstacle(1))^2 + (py - obstacle(2))^2);
end
terminal_heading_error = angle_error(heading, target_heading);
cost = cost + terminal_weight * ( ...
    position_weight * ((px - target_x)^2 + (py - target_y)^2) ...
    + heading_weight * terminal_heading_error^2);

problem.u = u;
problem.theta = theta;
problem.variable = variable;
problem.cost = cost;
problem.constraints = constraints;
problem.box_lower = repmat([-1.0; -deg2rad(25.0)], horizon, 1);
problem.box_upper = repmat([1.0; deg2rad(25.0)], horizon, 1);

meta.kind = 'circle';
meta.horizon = horizon;
meta.dt = dt;
meta.n = n;
meta.ncon = horizon;
meta.wheelbase = wheelbase;
meta.obstacle = obstacle;
meta.safe_radius = 0.4;
meta.start = [-1.2; 0.0; 0.0];
meta.target = [1.2; 0.0; 0.0];
meta.theta = [10.0; 0.2; 1e-2; 1e-2; 30.0; meta.safe_radius];
meta.variable = [meta.start; meta.target];
meta.theta_names = { ...
    'position_weight', 'heading_weight', 'speed_weight', 'steer_weight', ...
    'terminal_weight', 'safe_radius'};
meta.variable_names = {'start_x', 'start_y', 'start_heading', 'target_x', 'target_y', 'target_heading'};
meta.constraint_lower = -1e20 * ones(horizon, 1);
meta.constraint_upper = zeros(horizon, 1);
meta.initial_guess = bicycle_initial_guess(meta.start, meta.target, horizon, dt, wheelbase, 0.65, 1.0, deg2rad(25.0));
end

function [problem, meta] = build_rectangle_problem()
import casadi.*
horizon = 20;
dt = 0.12;
nu = 2;
n = horizon * nu;
u = SX.sym('u', n, 1);
theta = SX.sym('theta', 9, 1);
variable = SX.sym('variable', 6, 1);

position_weight = theta(1);
heading_weight = theta(2);
speed_weight = theta(3);
steer_weight = theta(4);
terminal_weight = theta(5);
margin_left = theta(6);
margin_right = theta(7);
margin_bottom = theta(8);
margin_top = theta(9);
px = variable(1);
py = variable(2);
heading = variable(3);
target_x = variable(4);
target_y = variable(5);
target_heading = variable(6);
xmin = -0.35 - margin_left;
xmax = 0.35 + margin_right;
ymin = -0.22 - margin_bottom;
ymax = 0.22 + margin_top;
wheelbase = 0.45;

cost = 0;
constraints = SX.zeros(horizon, 1);
for k = 1:horizon
    uk = u(2*k-1:2*k);
    [px, py, heading] = bicycle_step(px, py, heading, uk, dt, wheelbase);
    dx = px - target_x;
    dy = py - target_y;
    dheading = angle_error(heading, target_heading);
    cost = cost + position_weight * (dx^2 + dy^2) ...
        + heading_weight * dheading^2 ...
        + speed_weight * uk(1)^2 ...
        + steer_weight * uk(2)^2;
    constraints(k) = 0.5 ...
        * fmax(px - xmin, 0)^2 * fmax(xmax - px, 0)^2 ...
        * fmax(py - ymin, 0)^2 * fmax(ymax - py, 0)^2;
end
terminal_heading_error = angle_error(heading, target_heading);
cost = cost + terminal_weight * ( ...
    position_weight * ((px - target_x)^2 + (py - target_y)^2) ...
    + heading_weight * terminal_heading_error^2);

problem.u = u;
problem.theta = theta;
problem.variable = variable;
problem.cost = cost;
problem.constraints = constraints;
problem.box_lower = repmat([-1.0; -deg2rad(25.0)], horizon, 1);
problem.box_upper = repmat([1.0; deg2rad(25.0)], horizon, 1);

meta.kind = 'rectangle';
meta.horizon = horizon;
meta.dt = dt;
meta.n = n;
meta.ncon = horizon;
meta.wheelbase = wheelbase;
meta.rectangle_original = [-0.35; 0.35; -0.22; 0.22];
meta.rectangle = [-0.45; 0.45; -0.32; 0.32];
meta.start = [-1.2; 0.0; 0.0];
meta.target = [1.2; 0.0; 0.0];
meta.theta = [5.0; 0.2; 1e-2; 1e-2; 20.0; 0.1; 0.1; 0.1; 0.1];
meta.variable = [meta.start; meta.target];
meta.theta_names = { ...
    'position_weight', 'heading_weight', 'speed_weight', 'steer_weight', ...
    'terminal_weight', 'margin_left', 'margin_right', 'margin_bottom', 'margin_top'};
meta.variable_names = {'start_x', 'start_y', 'start_heading', 'target_x', 'target_y', 'target_heading'};
meta.constraint_lower = zeros(horizon, 1);
meta.constraint_upper = zeros(horizon, 1);
meta.initial_guess = bicycle_initial_guess(meta.start, meta.target, horizon, dt, wheelbase, 0.8, 1.0, deg2rad(25.0));
end

function [problem, meta] = build_quadratic_problem()
import casadi.*
horizon = 40;
dt = 0.1;
nu = 2;
n = horizon * nu;
u = SX.sym('u', n, 1);
theta = SX.sym('theta', 6, 1);
variable = SX.sym('variable', 6, 1);

position_weight = theta(1);
heading_weight = theta(2);
speed_weight = theta(3);
steer_weight = theta(4);
terminal_weight = theta(5);
safety_margin = theta(6);
px = variable(1);
py = variable(2);
heading = variable(3);
target_x = variable(4);
target_y = variable(5);
target_heading = variable(6);
wheelbase = 0.45;

cost = 0;
constraints = SX.zeros(horizon, 1);
for k = 1:horizon
    uk = u(2*k-1:2*k);
    [px, py, heading] = bicycle_step(px, py, heading, uk, dt, wheelbase);
    dx = px - target_x;
    dy = py - target_y;
    dheading = angle_error(heading, target_heading);
    cost = cost + position_weight * (dx^2 + dy^2) + heading_weight * dheading^2 ...
        + speed_weight * uk(1)^2 + steer_weight * uk(2)^2;
    lower_curve = px^2 - safety_margin;
    upper_curve = 1.0 + 0.5 * px^2 + safety_margin;
    h_lower = py - lower_curve;
    h_upper = upper_curve - py;
    constraints(k) = 0.5 * fmax(h_lower, 0)^2 * fmax(h_upper, 0)^2;
end
terminal_heading_error = angle_error(heading, target_heading);
cost = cost + terminal_weight * ( ...
    position_weight * ((px - target_x)^2 + (py - target_y)^2) ...
    + heading_weight * terminal_heading_error^2);

problem.u = u;
problem.theta = theta;
problem.variable = variable;
problem.cost = cost;
problem.constraints = constraints;
problem.box_lower = repmat([0.0; -deg2rad(30.0)], horizon, 1);
problem.box_upper = repmat([0.6; deg2rad(30.0)], horizon, 1);

meta.kind = 'quadratic';
meta.horizon = horizon;
meta.dt = dt;
meta.n = n;
meta.ncon = horizon;
meta.wheelbase = wheelbase;
meta.safety_margin = 0.15;
meta.start = [0.8; 0.0; pi];
meta.target = [-1.5; 1.0; -pi * 0.2];
meta.theta = [1.0; 0.01; 0.01; 0.01; 40.0; meta.safety_margin];
meta.variable = [meta.start; meta.target];
meta.theta_names = { ...
    'position_weight', 'heading_weight', 'speed_weight', 'steer_weight', ...
    'terminal_weight', 'safety_margin'};
meta.variable_names = {'start_x', 'start_y', 'start_heading', 'target_x', 'target_y', 'target_heading'};
meta.constraint_lower = zeros(horizon, 1);
meta.constraint_upper = zeros(horizon, 1);
meta.initial_guess = zeros(n, 1);
meta.rolling_start = [-0.2; 1.8; pi * 0.7];
meta.rolling_target = [-1.0; 0.5; -pi * 0.5];
meta.rolling_initial_guess = repmat([0.6; deg2rad(30.0)], horizon, 1);
end

function [px_next, py_next, heading_next] = bicycle_step(px, py, heading, control, dt, wheelbase)
speed = control(1);
steer = control(2);
px_next = px + dt * speed * cos(heading);
py_next = py + dt * speed * sin(heading);
heading_next = heading + dt * speed * tan(steer) / wheelbase;
end

function err = angle_error(a, b)
err = atan2(sin(a - b), cos(a - b));
end

function warm_start = bicycle_initial_guess(start, target, horizon, dt, wheelbase, y_amp, speed_limit, steer_limit)
waypoint_x = linspace(start(1), target(1), horizon + 1);
waypoint_y = y_amp * sin(linspace(0, pi, horizon + 1));
previous = start(1:2);
previous_heading = start(3);
warm_start = zeros(2 * horizon, 1);
for k = 1:horizon
    point = [waypoint_x(k + 1); waypoint_y(k + 1)];
    diff = point - previous;
    distance = norm(diff);
    if distance > 1e-12
        desired_heading = atan2(diff(2), diff(1));
    else
        desired_heading = previous_heading;
    end
    speed = min(max(distance / dt, -speed_limit), speed_limit);
    heading_rate = atan2(sin(desired_heading - previous_heading), cos(desired_heading - previous_heading)) / dt;
    if abs(speed) < 1e-8
        steer = 0.0;
    else
        steer = atan(wheelbase * heading_rate / speed);
    end
    warm_start(2*k-1:2*k) = [speed; min(max(steer, -steer_limit), steer_limit)];
    previous = point;
    previous_heading = desired_heading;
end
end
