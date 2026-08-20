function test_matlab_mex_workflow()
%TEST_MATLAB_MEX_WORKFLOW Smoke test for the MATLAB CasADi -> MEX workflow.
%
% Set CASADI_MATLAB_PATH before running this file if CasADi is not already
% on the MATLAB path, for example:
%   setenv('CASADI_MATLAB_PATH', 'C:\casadi-3.7.2-windows64-matlab2018b')
%   test_matlab_mex_workflow

setenv('CASADI_MATLAB_PATH', './casadi-3.7.2-windows64-matlab2018b');
casadi_path = getenv('CASADI_MATLAB_PATH');
if ~isempty(casadi_path)
    addpath(genpath(casadi_path));
end
addpath(fileparts(mfilename('fullpath')));

import casadi.*

u = SX.sym('u', 2, 1);
theta = SX.sym('theta', 2, 1);
variable = SX.sym('variable', 1, 1);

problem.u = u;
problem.theta = theta;
problem.variable = variable;
problem.cost = 0.5 * sumsqr(u - theta);
problem.outer_loss = sumsqr(u);
problem.constraints = u(1) + u(2);
problem.box_lower = [-1; -1];
problem.box_upper = [1; 1];

project_dir = fullfile(tempdir, 'lapanda_matlab_mex_smoke');
solver = lapanda_create_solver( ...
    problem, ...
    project_dir, ...
    'Name', 'lapanda_oracle', ...
    'MexName', 'lapanda_mex_smoke', ...
    'Force', false);

options.solver_max_iterations = 500;
options.solver_tolerance = 1e-8;
options.solver_buffer_size = 10;
options.solver_max_stable_iter = 80;
options.backward_enable = 1;
options.backward_tolerance = 1e-8;
options.backward_max_iterations = 500;
options.alm_max_iterations = 20;
options.alm_tolerance = 1e-6;
options.alm_initial_penalty = 10;
options.alm_penalty_update_factor = 2;
options.verbose = 0;

x0 = zeros(2, 1);
theta_val = [2.0; -0.25];
variable_val = 0.0;

panda_result = solver.solve_panda(x0, theta_val, variable_val, options);
expected_panda = [1.0; -0.25];
assert(norm(panda_result.solution - expected_panda, inf) < 1e-5);
assert(numel(panda_result.grad_theta) == 2);

theta_val = [0.8; -0.2];
constraint_lower = 0.0;
constraint_upper = 0.0;
alm_result = solver.solve_alm( ...
    x0, theta_val, variable_val, constraint_lower, constraint_upper, options);
expected_alm = [0.5; -0.5];
assert(norm(alm_result.solution - expected_alm, inf) < 1e-4);
assert(abs(sum(alm_result.solution)) < 1e-5);
assert(numel(alm_result.grad_theta) == 2);

disp('PANDA solution:');
disp(panda_result.solution);
disp('ALM solution:');
disp(alm_result.solution);
disp('MATLAB MEX workflow smoke test passed.');
end
