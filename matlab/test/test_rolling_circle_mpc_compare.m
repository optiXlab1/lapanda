function test_rolling_circle_mpc_compare()
%TEST_ROLLING_CIRCLE_MPC_COMPARE Rolling circle MPC: lapanda vs IPOPT.

import casadi.*

h = lapanda_test_helpers();
repo_root = h.setup_paths();
[problem, meta] = h.build_circle_problem();

alm_options = circle_alm_options();
project_dir = fullfile(tempdir, 'lapanda_matlab_roll_circle');
alm_solver = lapanda_create_solver(problem, project_dir, ...
    'Name', 'circle_rolling_oracle', ...
    'MexName', 'circle_rolling_mex', ...
    'Force', false);

u = vec(problem.u);
p = vertcat(vec(problem.theta), vec(problem.variable));
g = vec(problem.constraints);
ipopt_solver = nlpsol('ipopt_rolling_circle', 'ipopt', ...
    struct('x', u, 'p', p, 'f', problem.cost, 'g', g), ipopt_options());

out = run_rolling_compare(alm_solver, ipopt_solver, problem, meta, alm_options);
alm_metric = circle_metric(out.alm.trajectory(2:end, :), meta);
ipopt_metric = circle_metric(out.ipopt.trajectory(2:end, :), meta);

fprintf('\nMATLAB rolling circle MPC\n');
fprintf('lapanda: steps = %d, solve time = %.6f s, final distance = %.6e, min distance = %.6f\n', ...
    size(out.alm.controls, 2), out.alm.solve_time, alm_metric.final_distance, alm_metric.min_distance);
fprintf('IPOPT:     steps = %d, solve time = %.6f s, final distance = %.6e, min distance = %.6f\n', ...
    size(out.ipopt.controls, 2), out.ipopt.solve_time, ipopt_metric.final_distance, ipopt_metric.min_distance);

plot_path = plot_rolling_circle(repo_root, meta, out, 'rolling_circle_mpc_compare');
fprintf('plot path = %s\n', plot_path);

assert(alm_metric.final_distance < 0.15);
assert(ipopt_metric.final_distance < 0.15);
assert(alm_metric.min_distance >= meta.safe_radius - 2e-3);
assert(ipopt_metric.min_distance >= meta.safe_radius - 2e-3);
end

function out = run_rolling_compare(alm_solver, ipopt_solver, problem, meta, alm_options)
max_steps = 80;
target_tol = 0.15;
state_alm = meta.start;
state_ipopt = meta.start;
target = meta.target;
u0_alm = meta.initial_guess;
u0_ipopt = meta.initial_guess;
lambda0 = zeros(meta.ncon, 1);
lam_x0 = zeros(meta.n, 1);
lam_g0 = zeros(meta.ncon, 1);

out.alm.trajectory = state_alm.';
out.ipopt.trajectory = state_ipopt.';
out.alm.controls = [];
out.ipopt.controls = [];
out.alm.residuals = [];
out.alm.outer_iterations = [];
out.ipopt.iterations = [];

tic;
for k = 1:max_steps
    variable = [state_alm; target];
    alm_options.alm_multiplier0 = lambda0;
    r = alm_solver.solve_alm(u0_alm, meta.theta, variable, ...
        meta.constraint_lower, meta.constraint_upper, alm_options);
    control = r.solution(1:2);
    out.alm.controls(:, end + 1) = control;
    out.alm.residuals(end + 1) = r.final_residual;
    out.alm.outer_iterations(end + 1) = r.iterations;
    state_alm = state_alm + meta.dt * control;
    out.alm.trajectory(end + 1, :) = state_alm.';
    u0_alm = shift_controls(r.solution);
    lambda0 = shift_constraints(r.multipliers);
    if norm(state_alm(1:2) - target(1:2)) < target_tol
        break;
    end
end
out.alm.solve_time = toc;

tic;
for k = 1:max_steps
    variable = [state_ipopt; target];
    sol = ipopt_solver( ...
        'x0', u0_ipopt, ...
        'p', [meta.theta; variable], ...
        'lbx', problem.box_lower, ...
        'ubx', problem.box_upper, ...
        'lbg', meta.constraint_lower, ...
        'ubg', meta.constraint_upper, ...
        'lam_x0', lam_x0, ...
        'lam_g0', lam_g0);
    u_sol = full(sol.x);
    control = u_sol(1:2);
    out.ipopt.controls(:, end + 1) = control;
    stats = ipopt_solver.stats();
    out.ipopt.iterations(end + 1) = stats.iter_count;
    state_ipopt = state_ipopt + meta.dt * control;
    out.ipopt.trajectory(end + 1, :) = state_ipopt.';
    u0_ipopt = shift_controls(u_sol);
    lam_x0 = shift_controls(full(sol.lam_x));
    lam_g0 = shift_constraints(full(sol.lam_g));
    if norm(state_ipopt(1:2) - target(1:2)) < target_tol
        break;
    end
end
out.ipopt.solve_time = toc;
end

function options = circle_alm_options()
options.solver_max_iterations = 800;
options.solver_tolerance = 1e-3;
options.solver_buffer_size = 10;
options.solver_max_stable_iter = 80;
options.alm_max_iterations = 30;
options.alm_tolerance = 1e-3;
options.alm_initial_penalty = 2.0;
options.alm_penalty_update_factor = 10.0;
options.alm_sufficient_decrease_factor = 0.25;
options.alm_warm_start_inner = 1;
options.backward_enable = 0;
options.verbose = 0;
end

function options = ipopt_options()
options = struct;
options.print_time = false;
options.ipopt.print_level = 0;
options.ipopt.sb = 'yes';
options.ipopt.tol = 1e-3;
options.ipopt.constr_viol_tol = 1e-3;
options.ipopt.acceptable_tol = 1e-3;
options.ipopt.max_iter = 2000;
options.ipopt.warm_start_init_point = 'yes';
end

function metric = circle_metric(points, meta)
d = sqrt(sum((points - meta.obstacle.').^2, 2));
metric.min_distance = min(d);
metric.max_violation = max(meta.safe_radius - d);
metric.final_distance = norm(points(end, :)' - meta.target);
end

function shifted = shift_controls(values)
shifted = [values(3:end); values(end-1:end)];
end

function shifted = shift_constraints(values)
shifted = [values(2:end); values(end)];
end

function output_path = plot_rolling_circle(repo_root, meta, out, file_stem)
output_path = make_figure_path(repo_root, file_stem);
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');
t = linspace(0, 2*pi, 160);
fill(ax, meta.obstacle(1) + meta.safe_radius*cos(t), meta.obstacle(2) + meta.safe_radius*sin(t), ...
    [0.8392, 0.1882, 0.1529], 'FaceAlpha', 0.22, ...
    'EdgeColor', [0.6000, 0.1000, 0.1000], 'LineWidth', 1.4, 'DisplayName', 'safe obstacle');
plot(ax, out.alm.trajectory(:, 1), out.alm.trajectory(:, 2), 'o-', ...
    'Color', [0.1216, 0.4667, 0.7059], 'LineWidth', 2.0, 'MarkerSize', 4.0, 'DisplayName', 'lapanda');
plot(ax, out.ipopt.trajectory(:, 1), out.ipopt.trajectory(:, 2), 's-', ...
    'Color', [1.0000, 0.4980, 0.0549], 'LineWidth', 1.8, 'MarkerSize', 3.5, 'DisplayName', 'IPOPT');
plot(ax, meta.start(1), meta.start(2), 's', 'Color', [0.1725, 0.6275, 0.1725], ...
    'MarkerFaceColor', [0.1725, 0.6275, 0.1725], 'MarkerSize', 8.0, 'DisplayName', 'start');
plot(ax, meta.target(1), meta.target(2), 'p', 'Color', [1.0000, 0.4980, 0.0000], ...
    'MarkerFaceColor', [1.0000, 0.4980, 0.0000], 'MarkerSize', 11.0, 'DisplayName', 'target');
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x');
ylabel(ax, 'y');
title(ax, 'rolling circle MPC compare');
legend(ax, 'Location', 'best');
xlim(ax, [-1.5, 1.5]);
ylim(ax, [-0.9, 0.9]);
saveas(fig, output_path);
end

function output_path = make_figure_path(repo_root, file_stem)
figure_dir = fullfile(repo_root, 'matlab', 'test', 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end
output_path = fullfile(figure_dir, [file_stem '.png']);
end
