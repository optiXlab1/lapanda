function test_rectangle_lapanda()
%TEST_RECTANGLE_lapanda Rectangle obstacle solved by MATLAB MEX lapanda.

h = lapanda_test_helpers();
repo_root = h.setup_paths();
[problem, meta] = h.build_rectangle_problem();

options.solver_max_iterations = 2000;
options.solver_tolerance = 1e-5;
options.solver_buffer_size = 10;
options.solver_max_stable_iter = 0;
options.alm_max_iterations = 30;
options.alm_tolerance = 1e-5;
options.alm_initial_penalty = 1e4;
options.alm_penalty_update_factor = 10.0;
options.alm_sufficient_decrease_factor = 0.25;
options.alm_warm_start_inner = 1;
options.backward_enable = 0;
options.verbose = 0;

project_dir = fullfile(tempdir, 'lapanda_matlab_test_rectangle_lapanda');
solver = lapanda_create_solver(problem, project_dir, ...
    'Name', 'test_rectangle_lapanda_oracle', ...
    'MexName', 'test_rectangle_lapanda_mex', ...
    'Force', false);

tic;
result = solver.solve_alm( ...
    meta.initial_guess, meta.theta, meta.variable, ...
    meta.constraint_lower, meta.constraint_upper, options);
solve_time = toc;

trajectory = rollout_bicycle(meta.variable, result.solution, meta.dt, meta.wheelbase);
metric = rectangle_metric(trajectory(:, 1:2), meta);

fprintf('\nMATLAB lapanda rectangle obstacle\n');
fprintf('solve time = %.6f s\n', solve_time);
fprintf('outer = %d, residual = %.6e, penalty = %.6e\n', ...
    result.iterations, result.final_residual, result.penalty);
fprintf('max product = %.6e, max signed violation = %.6e, final distance = %.6e\n', ...
    metric.max_product, metric.max_signed_violation, metric.final_distance);

plot_path = plot_rectangle_result(repo_root, meta, trajectory, 'lapanda', 'rectangle_lapanda');
fprintf('plot path = %s\n', plot_path);

assert(metric.max_product < 1e-5);
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

function metric = rectangle_metric(points, meta)
rect = meta.rectangle;
x = points(:, 1);
y = points(:, 2);
product = 0.5 .* max(x - rect(1), 0).^2 .* max(rect(2) - x, 0).^2 ...
    .* max(y - rect(3), 0).^2 .* max(rect(4) - y, 0).^2;
signed = min([x - rect(1), rect(2) - x, y - rect(3), rect(4) - y], [], 2);
metric.max_product = max(product);
metric.max_signed_violation = max(signed);
metric.final_distance = norm(points(end, 1:2)' - meta.target(1:2));
end

function output_path = plot_rectangle_result(repo_root, meta, trajectory, method_name, file_stem)
output_path = make_figure_path(repo_root, file_stem);
fig = figure('Color', 'w');
ax = axes('Parent', fig);
hold(ax, 'on');
rect = meta.rectangle;
patch(ax, [rect(1), rect(2), rect(2), rect(1)], [rect(3), rect(3), rect(4), rect(4)], ...
    [0.8392, 0.1882, 0.1529], 'FaceAlpha', 0.18, ...
    'EdgeColor', [0.6000, 0.1000, 0.1000], 'LineWidth', 1.4, ...
    'LineStyle', '--', 'DisplayName', 'safety rectangle');
original = meta.rectangle_original;
patch(ax, [original(1), original(2), original(2), original(1)], ...
    [original(3), original(3), original(4), original(4)], ...
    [0.8392, 0.1882, 0.1529], 'FaceAlpha', 0.28, ...
    'EdgeColor', [0.8392, 0.1882, 0.1529], 'LineWidth', 1.0, ...
    'DisplayName', 'original rectangle');
plot(ax, trajectory(:, 1), trajectory(:, 2), 'o-', ...
    'Color', [0.1216, 0.4667, 0.7059], 'LineWidth', 2.0, ...
    'MarkerSize', 4.0, 'DisplayName', method_name);
plot_start_target(ax, meta);
finish_rectangle_plot(ax, [method_name ' rectangle obstacle']);
saveas(fig, output_path);
end

function plot_start_target(ax, meta)
plot(ax, meta.start(1), meta.start(2), 's', 'Color', [0.1725, 0.6275, 0.1725], ...
    'MarkerFaceColor', [0.1725, 0.6275, 0.1725], 'MarkerSize', 8.0, 'DisplayName', 'start');
plot(ax, meta.target(1), meta.target(2), 'p', 'Color', [1.0000, 0.4980, 0.0000], ...
    'MarkerFaceColor', [1.0000, 0.4980, 0.0000], 'MarkerSize', 11.0, 'DisplayName', 'target');
end

function finish_rectangle_plot(ax, plot_title)
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x');
ylabel(ax, 'y');
title(ax, plot_title);
legend(ax, 'Location', 'best');
xlim(ax, [-1.5, 1.5]);
ylim(ax, [-0.9, 0.9]);
end

function output_path = make_figure_path(repo_root, file_stem)
figure_dir = fullfile(repo_root, 'matlab', 'test', 'figures');
if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end
output_path = fullfile(figure_dir, [file_stem '.png']);
end
