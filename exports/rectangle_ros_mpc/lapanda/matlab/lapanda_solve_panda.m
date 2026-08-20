function result = lapanda_solve_panda(solver, x0, theta, variable, options)
%LAPANDA_SOLVE_PANDA Solve a box-constrained problem with PANDA.

if nargin < 5 || isempty(options)
    options = struct();
end
if nargin < 4 || isempty(variable)
    variable = zeros(solver.meta.nvar, 1);
end

result = feval(solver.mex_name, 'panda', solver.meta, x0, theta, variable, options);
end
