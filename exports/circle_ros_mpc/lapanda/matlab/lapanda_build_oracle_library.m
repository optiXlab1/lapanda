function library_path = lapanda_build_oracle_library(project_dir, name, cmake_generator)
%LAPANDA_BUILD_ORACLE_LIBRARY Build the generated CasADi oracle as a DLL/SO.

if nargin < 2 || isempty(name)
    name = 'lapanda_oracle';
end
if nargin < 3
    cmake_generator = '';
end

project_dir = char(project_dir);
generated_dir = fullfile(project_dir, 'generated');
build_dir = fullfile(generated_dir, 'build');
source_name = [name '.c'];
source_path = fullfile(generated_dir, source_name);
cmake_file = fullfile(generated_dir, 'CMakeLists.txt');

if ~exist(source_path, 'file')
    error('lapanda:build_oracle', 'Generated source %s does not exist.', source_name);
end

library_path = local_find_library(build_dir, name);
if ~isempty(library_path) && local_library_is_fresh(library_path, source_path)
    return;
end

fid = fopen(cmake_file, 'w');
if fid < 0
    error('lapanda:build_oracle', 'Failed to write CMakeLists.txt.');
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'cmake_minimum_required(VERSION 3.15)\n');
fprintf(fid, 'project(%s_oracle C)\n', name);
fprintf(fid, 'add_library(%s SHARED %s)\n', name, source_name);
fprintf(fid, 'set_target_properties(%s PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)\n', name);
clear cleanup;

configure_cmd = sprintf('cmake -S "%s" -B "%s"', generated_dir, build_dir);
if ~isempty(cmake_generator)
    configure_cmd = sprintf('%s -G "%s"', configure_cmd, cmake_generator);
end
status = system(configure_cmd);
if status ~= 0
    error('lapanda:build_oracle', 'CMake configure failed.');
end
status = system(sprintf('cmake --build "%s" --config Release', build_dir));
if status ~= 0
    error('lapanda:build_oracle', 'CMake build failed.');
end

library_path = local_find_library(build_dir, name);
if isempty(library_path)
    error('lapanda:build_oracle', 'Built oracle library was not found.');
end
end

function fresh = local_library_is_fresh(library_path, source_path)
library_info = dir(library_path);
source_info = dir(source_path);
fresh = ~isempty(library_info) ...
    && ~isempty(source_info) ...
    && library_info.datenum >= source_info.datenum;
end

function library_path = local_find_library(build_dir, name)
if ispc
    patterns = {[name '.dll'], ['lib' name '.dll']};
elseif ismac
    patterns = {['lib' name '.dylib']};
else
    patterns = {['lib' name '.so']};
end
library_path = '';
for i = 1:numel(patterns)
    items = dir(fullfile(build_dir, '**', patterns{i}));
    if ~isempty(items)
        library_path = fullfile(items(1).folder, items(1).name);
        return;
    end
end
end
