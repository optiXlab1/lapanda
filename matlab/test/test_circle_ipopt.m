function test_circle_ipopt()
%TEST_CIRCLE_IPOPT Circle obstacle solved by CasADi/IPOPT in MATLAB.

import casadi.*

h = lapanda_test_helpers();
repo_root = h.setup_paths();
[problem, meta] = h.build_circle_problem();

u = vec(problem.u);
p = vertcat(vec(problem.theta), vec(problem.variable));
g = vec(problem.constraints);
solver = nlpsol('ipopt_circle', 'ipopt', ...
    struct('x', u, 'p', p, 'f', problem.cost, 'g', g), ipopt_options());

tic;
sol = solver( ...
    'x0', meta.initial_guess, ...
    'p', [meta.theta; meta.variable], ...
    'lbx', problem.box_lower, ...
    'ubx', problem.box_upper, ...
    'lbg', meta.constraint_lower, ...
    'ubg', meta.constraint_upper);
solve_time = toc;

trajectory = rollout_circle(meta.variable, full(sol.x), meta.dt);
metric = circle_metric(trajectory, meta);
stats = solver.stats();

fprintf('\nMATLAB IPOPT circle obstacle\n');
fprintf('solve time = %.6f s\n', solve_time);
fprintf('iterations = %d, objective = %.6e\n', stats.iter_count, full(sol.f));
fprintf('min distance = %.6f, final distance = %.6e\n', ...
    metric.min_distance, metric.final_distance);

plot_path = plot_circle_result(repo_root, meta, trajectory, 'IPOPT', 'circle_ipopt');
fprintf('plot path = %s\n', plot_path);
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
end

function trajectory = rollout_circle(variable, solution, dt)
px = variable(1);
py = variable(2);
heading = variable(3);
wheelbase = 0.45;
controls = reshape(solution, 2, []).';
trajectory = zeros(size(controls, 1), 3);
for k = 1:size(controls, 1)
    speed = controls(k, 1);
    steer = controls(k, 2);
    px = px + dt * speed * cos(heading);
    py = py + dt * speed * sin(heading);
    heading = heading + dt * speed * tan(steer) / wheelbase;
    trajectory(k, :) = [px, py, heading];
end
end

function metric = circle_metric(points, meta)
positions = points(:, 1:2);
d = sqrt(sum((positions - meta.obstacle.').^2, 2));
metric.min_distance = min(d);
metric.max_violation = max(meta.safe_radius - d);
metric.final_distance = norm(positions(end, :)' - meta.target(1:2));
end

function output_path = plot_circle_result(repo_root, meta, trajectory, method_name, file_stem)
output_path = make_figure_path(repo_root, file_stem);
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');
t = linspace(0, 2*pi, 160);
fill(ax, ...
    meta.obstacle(1) + meta.safe_radius * cos(t), ...
    meta.obstacle(2) + meta.safe_radius * sin(t), ...
    [0.8392, 0.1882, 0.1529], ...
    'FaceAlpha', 0.22, 'EdgeColor', [0.6000, 0.1000, 0.1000], ...
    'LineWidth', 1.4, 'DisplayName', 'safe obstacle');
plot(ax, trajectory(:, 1), trajectory(:, 2), 'o-', ...
    'Color', [0.1216, 0.4667, 0.7059], ...
    'LineWidth', 2.0, 'MarkerSize', 4.0, ...
    'DisplayName', method_name);
plot(ax, meta.start(1), meta.start(2), 's', ...
    'Color', [0.1725, 0.6275, 0.1725], ...
    'MarkerFaceColor', [0.1725, 0.6275, 0.1725], ...
    'MarkerSize', 8.0, 'DisplayName', 'start');
plot(ax, meta.target(1), meta.target(2), 'p', ...
    'Color', [1.0000, 0.4980, 0.0000], ...
    'MarkerFaceColor', [1.0000, 0.4980, 0.0000], ...
    'MarkerSize', 11.0, 'DisplayName', 'target');
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x');
ylabel(ax, 'y');
title(ax, [method_name ' circle obstacle']);
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
