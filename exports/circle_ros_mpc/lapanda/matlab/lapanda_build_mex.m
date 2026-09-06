function mex_path = lapanda_build_mex(project_dir, mex_name)
%lapanda_BUILD_MEX Build the dynamic MATLAB MEX solver adapter.
%
% The generated oracle is not linked into this MEX. Build it separately with
% lapanda_build_oracle_library and pass the resulting library path through
% meta.library_path when calling the MEX function.

if nargin < 2 || isempty(mex_name)
    mex_name = 'lapanda_mex';
end

project_dir = char(project_dir);
root = fullfile(project_dir, 'lapanda');
mex_source = fullfile(root, 'matlab', 'lapanda_mex.c');
expected_mex = fullfile(project_dir, [mex_name '.' mexext]);
if ~exist(mex_source, 'file')
    mex_source = fullfile(fileparts(mfilename('fullpath')), 'lapanda_mex.c');
end

sources = {};
sources = [sources local_c_files(fullfile(root, 'panda'))]; %#ok<AGROW>
sources{end + 1} = fullfile(root, 'alm', 'alm.c');
sources{end + 1} = mex_source;

stamp_path = fullfile(project_dir, [mex_name '.mexbuild.mat']);
source_stamp = local_sources_stamp(sources);
if exist(expected_mex, 'file') && exist(stamp_path, 'file')
    try
        loaded = load(stamp_path, 'source_stamp');
        if isequal(loaded.source_stamp, source_stamp)
            mex_path = expected_mex;
            return;
        end
    catch
    end
end

include_args = { ...
    ['-I' fullfile(root, 'include')], ...
    ['-I' fullfile(root, 'panda')], ...
    ['-I' fullfile(root, 'alm')], ...
    ['-I' fullfile(root, 'globals')] ...
};

link_args = {};
if isunix && ~ismac
    link_args{end + 1} = '-ldl';
end

mex('-R2018a', include_args{:}, sources{:}, link_args{:}, '-output', mex_name);
mex_path = expected_mex;
save(stamp_path, 'source_stamp');
end

function files = local_c_files(folder)
items = dir(fullfile(folder, '*.c'));
files = cell(1, numel(items));
for i = 1:numel(items)
    files{i} = fullfile(items(i).folder, items(i).name);
end
end

function stamp = local_sources_stamp(sources)
stamp = cell(numel(sources), 2);
for i = 1:numel(sources)
    info = dir(sources{i});
    stamp{i, 1} = sources{i};
    if isempty(info)
        stamp{i, 2} = NaN;
    else
        stamp{i, 2} = info.datenum;
    end
end
end
