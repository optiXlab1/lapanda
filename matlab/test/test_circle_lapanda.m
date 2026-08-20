function test_circle_lapanda()
%TEST_CIRCLE_lapanda Circle obstacle solved by MATLAB MEX lapanda.

h = lapanda_test_helpers();
repo_root = h.setup_paths();
[problem, meta] = h.build_circle_problem();

options.solver_max_iterations = 800;
options.solver_tolerance = 1e-1;
options.solver_buffer_size = 10;
options.solver_max_stable_iter = 80;
options.alm_max_iterations = 30;
options.alm_tolerance = 1e-3;
options.alm_initial_penalty = 100;
options.alm_penalty_update_factor = 10.0;
options.alm_sufficient_decrease_factor = 0.25;
options.alm_warm_start_inner = 1;
options.backward_enable = 0;
options.verbose = 0;

project_dir = fullfile(tempdir, 'lapanda_matlab_test_circle_lapanda');
solver = lapanda_create_solver(problem, project_dir, ...
    'Name', 'test_circle_lapanda_oracle', ...
    'MexName', 'test_circle_lapanda_mex', ...
    'Force', false);

tic;
result = solver.solve_alm( ...
    meta.initial_guess, meta.theta, meta.variable, ...
    meta.constraint_lower, meta.constraint_upper, options);
solve_time = toc;

trajectory = rollout_circle(meta.variable, result.solution, meta.dt);
metric = circle_metric(trajectory, meta);

fprintf('\nMATLAB lapanda circle obstacle\n');
fprintf('solve time = %.6f s\n', solve_time);
fprintf('outer = %d, residual = %.6e, penalty = %.6e\n', ...
    result.iterations, result.final_residual, result.penalty);
fprintf('min distance = %.6f, final distance = %.6e\n', ...
    metric.min_distance, metric.final_distance);

plot_path = plot_circle_result(repo_root, meta, trajectory, 'lapanda', 'circle_lapanda');
fprintf('plot path = %s\n', plot_path);
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
