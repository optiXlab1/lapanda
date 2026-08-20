function solver = lapanda_create_solver(problem, project_dir, varargin)
%lapanda_CREATE_SOLVER Build a dynamic lapanda MEX solver from CasADi.
%
% solver = lapanda_create_solver(problem, project_dir)
% solver = lapanda_create_solver(..., 'Name', name, 'MexName', mex_name, ...)
% If project_dir is omitted or empty, a shared temp cache directory is used.
% By default Force=false, so identical problems reuse generated C, oracle
% library, and MEX artifacts when they already exist.
%
% The returned solver has:
%   solver.solve_panda(x0, theta, variable, options)
%   solver.solve_alm(x0, theta, variable, lower, upper, options)

opts = local_options(varargin{:});
if nargin < 2 || isempty(project_dir)
    project_dir = fullfile(tempdir, 'lapanda_matlab_cache');
end
project_dir = char(project_dir);

lapanda_codegen_matlab(problem, project_dir, opts.Name, opts.Force);
library_path = lapanda_build_oracle_library(project_dir, opts.Name, opts.CMakeGenerator);

metadata = load(fullfile(project_dir, 'generated', 'lapanda_metadata.mat'));
meta = metadata.meta;
meta.library_path = library_path;

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(project_dir);
mex_path = lapanda_build_mex(project_dir, opts.MexName);
addpath(project_dir);

solver.project_dir = project_dir;
solver.name = opts.Name;
solver.mex_name = opts.MexName;
solver.mex_path = mex_path;
solver.oracle_library = library_path;
solver.meta = meta;
solver.solve_panda = @(varargin) lapanda_solve_panda(solver, varargin{:});
solver.solve_alm = @(varargin) lapanda_solve_alm(solver, varargin{:});
solver.solve_lapanda = solver.solve_alm;
end

function opts = local_options(varargin)
opts.Name = 'lapanda_oracle';
opts.MexName = 'lapanda_mex';
opts.Force = false;
opts.CMakeGenerator = '';

if mod(numel(varargin), 2) ~= 0
    error('lapanda:create_solver', 'Options must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    key = varargin{i};
    value = varargin{i + 1};
    if ~ischar(key) && ~isstring(key)
        error('lapanda:create_solver', 'Option names must be strings.');
    end
    switch lower(char(key))
        case 'name'
            opts.Name = char(value);
        case 'mexname'
            opts.MexName = char(value);
        case 'force'
            opts.Force = logical(value);
        case 'cmakegenerator'
            opts.CMakeGenerator = char(value);
        otherwise
            error('lapanda:create_solver', 'Unknown option: %s.', char(key));
    end
end
end
