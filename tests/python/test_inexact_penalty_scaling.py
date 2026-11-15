"""Nonlinear active-constraint example with an analytic frozen sensitivity."""
import casadi as ca
import numpy as np
from lapanda import CasadiProblem,build_solver,BackwardOptions


def test_effective_multiplier_drift_and_recentering(tmp_path):
    x=ca.SX.sym('x',2);theta=ca.SX.sym('a',2);z=ca.SX.sym('target',2)
    problem=CasadiProblem(u=x,theta=theta,variable=z,cost=.5*ca.sumsqr(x-theta),
        constraints=ca.vertcat(ca.sumsqr(x)-1),outer_loss=.5*ca.sumsqr(x-z))
    solver=build_solver(problem,backend='compiled',cache_dir=tmp_path,name='inexact_circle')
    a=np.array([2.,1.]);normal=a/np.linalg.norm(a);point=(1+1e-6)*normal;target=np.array([.1,-.2])
    rho=1000.;scaled=100*rho;constraint=point@point-1
    eta=(np.linalg.norm(a)/(1+1e-6)-1)/2
    lam=eta-rho*constraint
    centered=eta-scaled*constraint
    def backward(multiplier):
        b=BackwardOptions();b.enable=True;b.system='subproblem';b.linear_solver='cg';b.tolerance=1e-10
        b.max_iterations=200;b.constraint_penalty_scale=100.
        return np.asarray(solver.backward_lapanda_at_solution(point,a,target,np.array([-1e20]),np.zeros(1),
            np.array([multiplier]),np.array([rho]),.01,backward_options=b)['grad_theta'])
    plain=backward(lam);corrected=backward(centered)
    eta_plain=lam+scaled*constraint
    M_plain=(1+2*eta_plain)*np.eye(2)+4*scaled*np.outer(point,point)
    M_centered=(1+2*eta)*np.eye(2)+4*scaled*np.outer(point,point)
    np.testing.assert_allclose(plain,np.linalg.solve(M_plain,point-target),atol=2e-10,rtol=2e-9)
    np.testing.assert_allclose(corrected,np.linalg.solve(M_centered,point-target),atol=2e-10,rtol=2e-9)
    local_kkt=(np.eye(2)-np.outer(normal,normal))@(point-target)/(1+2*eta)
    assert np.linalg.norm(plain-local_kkt)>100*np.linalg.norm(corrected-local_kkt)
    np.testing.assert_allclose(centered+scaled*constraint,eta,atol=1e-15)
