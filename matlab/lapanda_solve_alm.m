function result = lapanda_solve_alm(solver, x0, theta, variable, lower, upper, options)
%lapanda_SOLVE_ALM Solve a constrained problem with ALM+PANDA.
%
% Optional warm starts can be passed through options:
%   options.alm_multiplier0 = previous_multipliers;
%   options.alm_penalty0    = previous_penalties;
%
% The result struct includes:
%   result.solution, result.multipliers, result.penalties

if nargin < 7 || isempty(options)
    options = struct();
end
if nargin < 4 || isempty(variable)
    variable = zeros(solver.meta.nvar, 1);
end

result = feval(solver.mex_name, 'alm', solver.meta, x0, theta, variable, lower, upper, options);
end
