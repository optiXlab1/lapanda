% Minimal MATLAB example for lapanda.
%
% Run after adding lapanda/matlab and CasADi to the MATLAB path.

import casadi.*

repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo, 'matlab'));

u = SX.sym('u', 2, 1);
theta = SX.sym('theta', 2, 1);
variable = SX.sym('variable', 2, 1);

problem.u = u;
problem.theta = theta;
problem.variable = variable;
problem.cost = 0.5 * sumsqr(u - theta) + 0.05 * sumsqr(u);
problem.outer_loss = 0.5 * sumsqr(u - variable);
problem.constraints = [u(1)^2 + u(2)^2 - 1.0; u(1) + u(2) - 0.4];
problem.box_lower = [-1.5; -1.5];
problem.box_upper = [1.5; 1.5];

solver = lapanda_create_solver( ...
    problem, ...
    fullfile(tempdir, 'lapanda_matlab_minimal'), ...
    'Name', 'lapanda_matlab_minimal', ...
    'MexName', 'lapanda_matlab_minimal_mex');

options.inner_max_iterations = 1000;
options.inner_tolerance = 1e-5;
options.inner_buffer_size = 10;
options.inner_max_stable_iter = 80;
options.alm_max_iterations = 20;
options.alm_tolerance = 1e-5;
options.alm_initial_penalty = 10.0;
options.alm_penalty_update_factor = 5.0;
options.alm_sufficient_decrease_factor = 0.25;
options.backward_enable = true;
options.backward_tolerance = 1e-5;
options.backward_max_iterations = 200;

x0 = [0.0; 0.0];
theta_value = [0.9; 0.8];
variable_value = [0.2; 0.1];
constraint_lower = [-1e20; -1e20];
constraint_upper = [0.0; 0.0];

result = solver.solve_lapanda( ...
    x0, theta_value, variable_value, constraint_lower, constraint_upper, options);

disp(result.solution);
disp(result.grad_theta);
