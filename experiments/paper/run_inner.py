import argparse
import json
from pathlib import Path

import casadi as ca
import pandas as pd
from threadpoolctl import threadpool_limits
from lapanda import SolverOptions,AlmOptions,BackwardOptions
from experiments.exp2_OCPs import utils
from .common import HERE,RESULTS,RUNS,np,flat
import time


def main():
    parser=argparse.ArgumentParser(description='PANDA/PANOC inner-solver ablation in the same ALM framework.')
    parser.add_argument('--models',nargs='+',default=['cartpole','quadrotor','robot_arm'])
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,default=RUNS/'inner_solvers')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];rng=np.random.default_rng(2026111403)
    with threadpool_limits(1):
        for model in a.models:
            ins=utils.build_instance(model,20)
            solver=utils.build_lapanda_solver(ins,model,'inner_'+model,'compiled',False)
            tol=1e-5 if model=='robot_arm' else 1e-4
            cfg=json.loads((HERE/'configs'/f'{model}_closed.json').read_text())
            ao=utils.alm_options(AlmOptions,20,tol,cfg['alm_max_penalty'],
                initial_penalty=cfg['alm_initial_penalty'],penalty_update_factor=cfg['alm_penalty_update_factor'])
            off=BackwardOptions();off.enable=False
            c=ins.problem.constraints;y=ca.SX.sym('y',c.numel())
            cf=ca.Function('constraint',[ins.problem.u,ins.problem.theta,ins.problem.variable],[c])
            gf=ca.Function('stationarity',[ins.problem.u,ins.problem.theta,ins.problem.variable,y],
                [ca.gradient(ins.problem.cost+ca.dot(y,c),ins.problem.u)])
            lower,upper=flat(ins.problem.box_lower),flat(ins.problem.box_upper)
            for protocol,epochs,indices in [('open',[0,400,800],[0,16,31]),('closed',[0,250,500],[1,25,49])]:
                for epoch in epochs:
                    with np.load(RESULTS/'learning'/model/protocol/'lapanda-alm'/f'epoch{epoch}.npz') as saved:
                        data={k:saved[k].copy() for k in saved.files}
                    for index in indices:
                        theta,variable=data['theta'][index],data['variable'][index]
                        for mode in ['warm','cold']:
                            initial=data['warm'][index].copy() if mode=='warm' else np.zeros_like(data['warm'][index])
                            multiplier=data['multipliers'][index-1].copy() if mode=='warm' and protocol=='closed' else np.zeros_like(data['multipliers'][index])
                            for round_number in range(a.rounds):
                                for repeat in range(a.repeats+1):
                                    for stable in rng.permutation([120,0]):
                                        io=utils.inner_options(SolverOptions,1500,tol);io.max_stable_iter=int(stable)
                                        start=time.perf_counter()
                                        result=solver.solve_lapanda(initial.copy(),theta.copy(),variable.copy(),ins.constraint_lower,ins.constraint_upper,
                                            inner_solver_options=io,alm_options=ao,backward_options=off,multiplier0=multiplier.copy())
                                        elapsed=(time.perf_counter()-start)*1e3
                                        if repeat==0:continue
                                        x,lam,rho=(flat(result[k]) for k in ['solution','multipliers','penalties'])
                                        cc=flat(cf(x,theta,variable))
                                        residual=cc-np.clip(cc+lam/rho,ins.constraint_lower,ins.constraint_upper)
                                        gradient=flat(gf(x,theta,variable,lam+rho*residual))
                                        station=float(np.max(abs(x-np.clip(x-gradient,lower,upper))))
                                        violation=float(np.max(np.maximum(ins.constraint_lower-cc,np.maximum(cc-ins.constraint_upper,0))))
                                        rows.append(dict(model=model,protocol=protocol,epoch=epoch,sample=index,mode=mode,
                                            method='PANDA' if stable else 'PANOC',round=round_number,repeat=repeat,
                                            wall_ms=elapsed,iterations=int(sum(result['inner_iterations'])),
                                            feasibility=violation,stationarity=station,
                                            passed=bool(max(station,violation)<=1.01*tol)))
                    pd.DataFrame(rows).to_csv(a.output/'raw.csv',index=False)
            print('Completed',model,flush=True)
    frame=pd.DataFrame(rows)
    keys=['model','protocol','epoch','sample','mode','method']
    points=frame.groupby(keys,as_index=False).agg(wall_ms=('wall_ms','median'),iterations=('iterations','median'),passed=('passed','all'))
    points.to_csv(a.output/'points.csv',index=False)
    paired=points.groupby(keys[:-1],as_index=False).agg(both_pass=('passed','all'))
    eligible=points.merge(paired[paired.both_pass].drop(columns='both_pass'),on=keys[:-1])
    summary=eligible.groupby(['model','mode','method'],as_index=False).agg(
        mean_wall_ms=('wall_ms','mean'),mean_iterations=('iterations','mean'),points=('passed','size'))
    summary.to_csv(a.output/'summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':main()
