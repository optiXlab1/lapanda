function test_quadratic_band_ipopt()
%TEST_QUADRATIC_BAND_IPOPT Nonconvex quadratic band solved by CasADi/IPOPT in MATLAB.

import casadi.*

h = lapanda_test_helpers();
repo_root = h.setup_paths();
[problem, meta] = h.build_quadratic_problem();

u = vec(problem.u);
p = vertcat(vec(problem.theta), vec(problem.variable));
g = vec(problem.constraints);
solver = nlpsol('ipopt_quadratic_band', 'ipopt', ...
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

trajectory = rollout_bicycle(meta.variable, full(sol.x), meta.dt, meta.wheelbase);
metric = quadratic_metric(trajectory(:, 1:2), meta);
stats = solver.stats();

fprintf('\nMATLAB IPOPT quadratic-band obstacle\n');
fprintf('solve time = %.6f s\n', solve_time);
fprintf('iterations = %d, objective = %.6e\n', stats.iter_count, full(sol.f));
fprintf('max product = %.6e, max signed violation = %.6e, final distance = %.6e\n', ...
    metric.max_product, metric.max_signed_violation, metric.final_distance);

plot_path = plot_quadratic_result(repo_root, meta, trajectory, 'IPOPT', 'quadratic_band_ipopt');
fprintf('plot path = %s\n', plot_path);

assert(metric.max_product < 1e-3);
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

function trajectory = rollout_bicycle(variable, solution, dt, wheelbase)
px = variable(1);
py = variable(2);
heading = variable(3);
controls = reshape(solution, 2, []).';
trajectory = zeros(size(controls, 1), 3);
for k = 1:size(controls, 1)
    speed = controls(k, 1);
    steer = controls(k, 2);
    px = px + dt * speed * cos(heading);
    py = py + dt * speed * sin(heading);
    heading = heading + dt * speed * tan(steer) / wheelbase;
    heading = atan2(sin(heading), cos(heading));
    trajectory(k, :) = [px, py, heading];
end
end

function metric = quadratic_metric(points, meta)
x = points(:, 1);
y = points(:, 2);
lower_curve = x.^2 - meta.safety_margin;
upper_curve = 1.0 + 0.5 .* x.^2 + meta.safety_margin;
h_lower = y - lower_curve;
h_upper = upper_curve - y;
product = 0.5 .* max(h_lower, 0).^2 .* max(h_upper, 0).^2;
signed = min(h_lower, h_upper);
metric.max_product = max(product);
metric.max_signed_violation = max(signed);
metric.final_distance = norm(points(end, 1:2)' - meta.target(1:2));
end

function output_path = plot_quadratic_result(repo_root, meta, trajectory, method_name, file_stem)
output_path = make_figure_path(repo_root, file_stem);
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');
x = linspace(-1.8, 1.8, 240);
lower_safe = x.^2 - meta.safety_margin;
upper_safe = 1.0 + 0.5 .* x.^2 + meta.safety_margin;
fill(ax, [x, fliplr(x)], [lower_safe, fliplr(upper_safe)], ...
    [0.8392, 0.1882, 0.1529], 'FaceAlpha', 0.20, ...
    'EdgeColor', [0.6000, 0.1000, 0.1000], 'LineWidth', 1.2, ...
    'DisplayName', 'safety band');
plot(ax, x, x.^2, '--', 'Color', [0.6000, 0.1000, 0.1000], ...
    'LineWidth', 1.0, 'DisplayName', 'original band');
plot(ax, x, 1.0 + 0.5 .* x.^2, '--', 'Color', [0.6000, 0.1000, 0.1000], ...
    'LineWidth', 1.0, 'HandleVisibility', 'off');
plot(ax, trajectory(:, 1), trajectory(:, 2), 'o-', ...
    'Color', [0.1216, 0.4667, 0.7059], 'LineWidth', 2.0, ...
    'MarkerSize', 4.0, 'DisplayName', method_name);
plot_start_target(ax, meta);
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x');
ylabel(ax, 'y');
title(ax, [method_name ' quadratic-band obstacle']);
legend(ax, 'Location', 'best');
xlim(ax, [-2.0, 1.8]);
ylim(ax, [-0.8, 3.0]);
saveas(fig, output_path);
end

function plot_start_target(ax, meta)
plot(ax, meta.start(1), meta.start(2), 's', 'Color', [0.1725, 0.6275, 0.1725], ...
    'MarkerFaceColor', [0.1725, 0.6275, 0.1725], 'MarkerSize', 8.0, 'DisplayName', 'start');
plot(ax, meta.target(1), meta.target(2), 'p', 'Color', [1.0000, 0.4980, 0.0000], ...
    'MarkerFaceColor', [1.0000, 0.4980, 0.0000], 'MarkerSize', 11.0, 'DisplayName', 'target');
end

function output_path = make_figure_path(repo_root, file_stem)
figure_dir = fullfile(repo_root, 'matlab', 'test', 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end
output_path = fullfile(figure_dir, [file_stem '.png']);
end
