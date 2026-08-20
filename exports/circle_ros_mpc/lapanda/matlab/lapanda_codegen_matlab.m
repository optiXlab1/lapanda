function out_dir = lapanda_codegen_matlab(problem, out_dir, name, force)
%LAPANDA_CODEGEN_MATLAB Generate CasADi C oracle files for lapanda.
%
% Required fields:
%   problem.u       CasADi decision variable
%   problem.theta   CasADi parameter vector
%   problem.cost    smooth inner objective
%
% Optional fields:
%   problem.variable    extra runtime data vector, use [] if unused
%   problem.outer_loss  outer loss for backward
%   problem.constraints c(u,theta,variable) for ALM
%   problem.box_lower   PANDA box lower bound
%   problem.box_upper   PANDA box upper bound

import casadi.*

if nargin < 3 || isempty(name)
    name = 'lapanda_oracle';
end
if nargin < 4
    force = false;
end

out_dir = char(out_dir);
generated_dir = fullfile(out_dir, 'generated');
fingerprint = local_problem_fingerprint(problem, name);
metadata_path = fullfile(generated_dir, 'lapanda_metadata.mat');
source_path = fullfile(generated_dir, [name '.c']);
header_path = fullfile(generated_dir, [name '.h']);

if ~force && local_cache_is_valid(metadata_path, source_path, header_path, out_dir, fingerprint)
    return;
end

if force && exist(out_dir, 'dir')
    rmdir(out_dir, 's');
end
if ~exist(generated_dir, 'dir')
    mkdir(generated_dir);
end
local_copy_sources(out_dir);

u = vec(problem.u);
theta = vec(problem.theta);
if isfield(problem, 'variable') && ~isempty(problem.variable)
    variable = vec(problem.variable);
else
    variable = SX.sym('variable', 0, 1);
end

n = local_numel(u);
ntheta = local_numel(theta);
nvar = local_numel(variable);
v = SX.sym('v', n, 1);

cost = problem.cost;
grad = gradient(cost, u);
functions = {};
functions{end + 1} = Function('panda_cost_grad', {u, theta, variable}, {cost, densify(grad)});
functions{end + 1} = Function('panda_hvp', {v, u, theta, variable}, {densify(jtimes(grad, u, v))});
functions{end + 1} = Function('panda_vjp', {v, u, theta, variable}, {densify(jacobian(grad, theta).' * v)});

has_outer_loss = isfield(problem, 'outer_loss') && ~isempty(problem.outer_loss);
if has_outer_loss
    loss_grad = gradient(problem.outer_loss, u);
    functions{end + 1} = Function( ...
        'panda_loss_grad', {u, theta, variable}, {problem.outer_loss, densify(loss_grad)});
end

ncon = 0;
if isfield(problem, 'constraints') && ~isempty(problem.constraints)
    c = vec(problem.constraints);
    ncon = local_numel(c);
    weight = SX.sym('weight', ncon, 1);
    jtprod = jacobian(c, u).' * weight;
    functions{end + 1} = Function('panda_constraints', {u, theta, variable}, {c});
    functions{end + 1} = Function( ...
        'panda_constraint_jtprod', {u, theta, variable, weight}, {densify(jtprod)});
    functions{end + 1} = Function( ...
        'panda_constraint_jv', {v, u, theta, variable}, {densify(jtimes(c, u, v))});
    functions{end + 1} = Function( ...
        'panda_constraint_weighted_hvp', {v, u, theta, variable, weight}, ...
        {densify(jtimes(jtprod, u, v))});
    functions{end + 1} = Function( ...
        'panda_constraint_weighted_vjp', {v, u, theta, variable, weight}, ...
        {densify(jacobian(jtprod, theta).' * v)});
    functions{end + 1} = Function( ...
        'panda_constraint_theta_jtprod', {u, theta, variable, weight}, ...
        {densify(jacobian(c, theta).' * weight)});
end

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(generated_dir);
generator = CodeGenerator([name '.c'], struct('with_header', true));
for i = 1:numel(functions)
    generator.add(functions{i});
end
generator.generate();

box_lower = [];
box_upper = [];
if isfield(problem, 'box_lower') && ~isempty(problem.box_lower)
    box_lower = full(problem.box_lower(:)).';
end
if isfield(problem, 'box_upper') && ~isempty(problem.box_upper)
    box_upper = full(problem.box_upper(:)).';
end
local_write_config( ...
    fullfile(generated_dir, 'lapanda_generated_config.h'), ...
    name, n, ntheta, nvar, ncon, box_lower, box_upper, has_outer_loss);
meta.name = name;
meta.n = n;
meta.ntheta = ntheta;
meta.nvar = nvar;
meta.ncon = ncon;
meta.box_lower = box_lower(:);
meta.box_upper = box_upper(:);
meta.has_outer_loss = has_outer_loss;
meta.fingerprint = fingerprint;
meta.library_path = '';
save(metadata_path, 'meta');
end

function ok = local_cache_is_valid(metadata_path, source_path, header_path, out_dir, fingerprint)
ok = false;
if ~exist(metadata_path, 'file') || ~exist(source_path, 'file') || ~exist(header_path, 'file')
    return;
end
if ~exist(fullfile(out_dir, 'lapanda', 'matlab', 'lapanda_mex.c'), 'file')
    return;
end
try
    loaded = load(metadata_path, 'meta');
    ok = isfield(loaded, 'meta') ...
        && isfield(loaded.meta, 'fingerprint') ...
        && strcmp(char(loaded.meta.fingerprint), char(fingerprint));
catch
    ok = false;
end
end

function fingerprint = local_problem_fingerprint(problem, name)
pieces = {
    name
    local_expr_text(problem.u)
    local_expr_text(problem.theta)
    local_optional_expr(problem, 'variable')
    local_expr_text(problem.cost)
    local_optional_expr(problem, 'constraints')
    local_optional_expr(problem, 'outer_loss')
    local_optional_numeric(problem, 'box_lower')
    local_optional_numeric(problem, 'box_upper')
};
fingerprint = local_sha256(strjoin(pieces, char(0)));
fingerprint = fingerprint(1:16);
end

function text = local_optional_expr(problem, field)
if isfield(problem, field) && ~isempty(problem.(field))
    text = local_expr_text(problem.(field));
else
    text = '<empty>';
end
end

function text = local_optional_numeric(problem, field)
if isfield(problem, field) && ~isempty(problem.(field))
    value = full(problem.(field)(:));
    parts = cell(numel(value), 1);
    for i = 1:numel(value)
        parts{i} = sprintf('%.17g', value(i));
    end
    text = strjoin(parts, ',');
else
    text = '<empty>';
end
end

function text = local_expr_text(expr)
try
    text = char(expr);
catch
    text = evalc('disp(expr)');
end
end

function hex = local_sha256(text)
md = java.security.MessageDigest.getInstance('SHA-256');
bytes = md.digest(uint8(text));
hex_chars = lower(dec2hex(typecast(bytes, 'uint8'))).';
hex = hex_chars(:).';
end

function n = local_numel(x)
s = size(x);
n = prod(s);
end

function local_copy_sources(out_dir)
repo_root = fileparts(fileparts(mfilename('fullpath')));
dst_root = fullfile(out_dir, 'lapanda');
items = {'panda', 'alm', 'include', 'globals', 'adapters', 'matlab'};
if ~exist(dst_root, 'dir')
    mkdir(dst_root);
end
for i = 1:numel(items)
    src = fullfile(repo_root, items{i});
    dst = fullfile(dst_root, items{i});
    if exist(dst, 'dir')
        rmdir(dst, 's');
    end
    if strcmp(items{i}, 'matlab')
        mkdir(dst);
        local_copy_pattern(src, dst, '*.m');
        local_copy_pattern(src, dst, '*.c');
    else
        copyfile(src, dst);
    end
end
end

function local_copy_pattern(src, dst, pattern)
items = dir(fullfile(src, pattern));
for i = 1:numel(items)
    copyfile(fullfile(items(i).folder, items(i).name), fullfile(dst, items(i).name));
end
end

function local_write_config(path, name, n, ntheta, nvar, ncon, box_lower, box_upper, has_outer_loss)
fid = fopen(path, 'w');
if fid < 0
    error('lapanda:codegen', 'Failed to open generated config file.');
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '#ifndef LAPANDA_GENERATED_CONFIG_H\n');
fprintf(fid, '#define LAPANDA_GENERATED_CONFIG_H\n\n');
fprintf(fid, '#include "%s.h"\n\n', name);
fprintf(fid, '#define LAPANDA_N %d\n', n);
fprintf(fid, '#define LAPANDA_NTHETA %d\n', ntheta);
fprintf(fid, '#define LAPANDA_NVAR %d\n', nvar);
fprintf(fid, '#define LAPANDA_NCON %d\n', ncon);
fprintf(fid, '#define LAPANDA_HAS_BOX_LOWER %d\n', ~isempty(box_lower));
fprintf(fid, '#define LAPANDA_HAS_BOX_UPPER %d\n', ~isempty(box_upper));
fprintf(fid, '#define LAPANDA_HAS_OUTER_LOSS %d\n\n', has_outer_loss);
fprintf(fid, 'static const double LAPANDA_BOX_LOWER[LAPANDA_N] = %s;\n', local_c_array(box_lower));
fprintf(fid, 'static const double LAPANDA_BOX_UPPER[LAPANDA_N] = %s;\n\n', local_c_array(box_upper));
fprintf(fid, '#endif\n');
end

function text = local_c_array(values)
if isempty(values)
    text = '{0.0}';
    return;
end
parts = cell(1, numel(values));
for i = 1:numel(values)
    parts{i} = sprintf('%.17g', values(i));
end
text = ['{' strjoin(parts, ', ') '}'];
end
