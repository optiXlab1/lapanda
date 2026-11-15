import argparse
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.exp1_rosenbrock_smooth_constraints.problem import build_problem
from experiments.exp2_OCPs import utils
from .common import HERE, RESULTS, RUNS, np
from .c_review_linear import CReview


def main():
    parser=argparse.ArgumentParser(description='Preconditioned and unpreconditioned backward solves.')
    parser.add_argument('--study',choices=['cg','minres'],required=True)
    parser.add_argument('--models',nargs='+',default=['cartpole','quadrotor','robot_arm'])
    parser.add_argument('--sizes',nargs='+',type=int,default=[100,200,500,1000])
    parser.add_argument('--tolerances',nargs='+',type=float)
    parser.add_argument('--repeats',type=int,default=11)
    parser.add_argument('--memory',action='store_true')
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    out=a.output or RUNS/'preconditioning'/a.study
    out.mkdir(parents=True,exist_ok=True)
    tolerances=a.tolerances or ([1e-3,1e-6,1e-8] if a.study=='cg' else [1e-2,1e-3,1e-6])
    rng=np.random.default_rng(20261114)
    rows=[]
    memory=[]
    with threadpool_limits(1):
        for key in (a.models if a.study=='cg' else a.sizes):
            ins=utils.build_instance(key,20) if a.study=='cg' else build_problem(key)
            methods=['cg','jacobi_pcg','lbfgs_pcg'] if a.study=='cg' else ['minres','pminres']
            backend=CReview('precondition_'+str(key),ins,methods[0],1e-3,forward_solver=False,
                            kernel_path=HERE/'c_review_efficient.c',schur_diagonal=True,max_iterations=10000)
            paths=sorted((RESULTS/'preconditioning/cg/states').glob(str(key)+'_*.npz')) if a.study=='cg' else [
                RESULTS/'rosenbrock/main/states'/f'n{key}_seed{seed}.npz' for seed in range(107,112)]
            for path in paths:
                with np.load(path) as saved:state={k:saved[k].copy() for k in saved.files}
                for tol in tolerances:
                    backend.tol=tol
                    for method in methods:
                        backend.review_method=method
                        backend.backward(state,state['theta'],state['variable'])
                    for repeat in range(a.repeats):
                        for method in rng.permutation(methods):
                            backend.review_method=method
                            _,info=backend.backward(state,state['theta'],state['variable'])
                            rows.append(dict(problem=key,case=path.stem,method=method,tolerance=tol,
                                repeat=repeat,wall_us=1e6*info['wall_sec'],iterations=info['iterations'],
                                residual=info['residual'],status=info['status']))
                    if a.memory:
                        import memray
                        for method in methods:
                            backend.review_method=method
                            for repeat in range(3):
                                capture=out/f'memory_{path.stem}_{method}_{tol}_{repeat}.bin'
                                with memray.Tracker(str(capture),native_traces=True,trace_python_allocators=True):
                                    _,info=backend.backward(state,state['theta'],state['variable'])
                                reader=memray.FileReader(str(capture))
                                peak=sum(r.size for r in reader.get_high_watermark_allocation_records(merge_threads=True))
                                memory.append(dict(problem=key,case=path.stem,method=method,tolerance=tol,
                                    repeat=repeat,peak_allocation_bytes=peak,residual=info['residual'],status=info['status']))
                        pd.DataFrame(memory).to_csv(out/'memory.csv',index=False)
                pd.DataFrame(rows).to_csv(out/'raw.csv',index=False)
            print('Completed',key,flush=True)
    frame=pd.DataFrame(rows)
    points=frame.groupby(['problem','case','method','tolerance'],as_index=False).agg(
        wall_us=('wall_us','median'),iterations=('iterations','median'),
        residual=('residual','max'),status=('status','max'))
    points.to_csv(out/'points.csv',index=False)
    summary=points.groupby(['problem','method','tolerance'],as_index=False).agg(
        median_wall_us=('wall_us','median'),median_iterations=('iterations','median'),
        maximum_residual=('residual','max'),failures=('status',lambda s:int((s!=0).sum())))
    summary.to_csv(out/'summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':main()
