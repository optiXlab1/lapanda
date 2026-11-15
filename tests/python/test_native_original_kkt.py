"""Native public API: analytic original derivatives and unchanged forward path."""
import casadi as ca
import numpy as np
import pytest
from lapanda import CasadiProblem,build_solver,BackwardOptions,SolverOptions,AlmOptions


def options(system='original_kkt',scale=1.):
    b=BackwardOptions();b.enable=True;b.system=system;b.linear_solver='minres' if system=='original_kkt' else 'cg'
    b.tolerance=1e-11;b.max_iterations=200;b.constraint_penalty_scale=scale
    return b


def equality_problem(tmp_path,fixed_box=False):
    x=ca.SX.sym('x',2);t=ca.SX.sym('theta');z=ca.SX.sym('z')
    p=CasadiProblem(u=x,theta=t,variable=z,cost=.5*(x[0]**2+4*x[1]**2),
        constraints=ca.vertcat(x[0]+x[1]-t),outer_loss=.5*ca.sumsqr(x),
        box_lower=np.array([.3,-10.]) if fixed_box else None,
        box_upper=np.array([.3,10.]) if fixed_box else None)
    return build_solver(p,backend='compiled',cache_dir=tmp_path,name='native_eq')


def backward(solver,x,lam,rho,b,theta=.8):
    return solver.backward_lapanda_at_solution(np.array(x),np.array([theta]),np.zeros(1),
        np.zeros(1),np.zeros(1),np.array([lam]),np.array([rho]),.1,backward_options=b)


def test_exact_original_gradient_and_finite_penalty_proxy(tmp_path):
    solver=equality_problem(tmp_path)
    for rho in [1.,10.,1000.]:
        kkt=backward(solver,[.64,.16],-.64,rho,options())
        cg=backward(solver,[.64,.16],-.64,rho,options('subproblem'))
        np.testing.assert_allclose(kkt['grad_theta'],[.544],atol=2e-10)
        np.testing.assert_allclose(cg['grad_theta'],[.544*rho/(rho+.8)],atol=2e-10)
        assert kkt['backward_system_dimension']==3
        assert kkt['backward_residual']<=1e-11
    scaled=backward(solver,[.64,.16],-.64,10.,options('subproblem',100.))
    np.testing.assert_allclose(scaled['grad_theta'],[.544*1000/1000.8],atol=2e-10)


def test_original_kkt_constant_box_face(tmp_path):
    solver=equality_problem(tmp_path,True)
    result=backward(solver,[.3,.5],-2.,10.,options())
    np.testing.assert_allclose(result['grad_theta'],[.5],atol=2e-10)
    assert result['backward_free_variables']==1
    assert result['backward_system_dimension']==2


def test_native_forward_is_identical_for_both_systems(tmp_path):
    solver=equality_problem(tmp_path)
    io=SolverOptions();io.tolerance=1e-10;io.max_iterations=2000
    ao=AlmOptions();ao.tolerance=1e-8;ao.max_iterations=30;ao.initial_penalty=10.;ao.penalty_update_factor=5.
    kwargs=dict(x0=np.zeros(2),theta=np.array([.8]),variable=np.zeros(1),
        constraint_lower=np.zeros(1),constraint_upper=np.zeros(1),inner_solver_options=io,alm_options=ao)
    cg=solver.solve_lapanda(**kwargs,backward_options=options('subproblem'))
    kkt=solver.solve_lapanda(**kwargs,backward_options=options())
    np.testing.assert_array_equal(cg['solution'],kkt['solution'])
    np.testing.assert_array_equal(cg['multipliers'],kkt['multipliers'])
    np.testing.assert_array_equal(cg['penalties'],kkt['penalties'])
    np.testing.assert_allclose(kkt['grad_theta'],[.544],atol=2e-7)
    assert kkt['backward_system']=='original_kkt'


def test_deferred_backward_preserves_gradient_and_next_forward(tmp_path):
    solver=equality_problem(tmp_path)
    io=SolverOptions();io.tolerance=1e-10;io.max_iterations=2000
    ao=AlmOptions();ao.tolerance=1e-8;ao.max_iterations=30;ao.initial_penalty=10.;ao.penalty_update_factor=5.
    kwargs=dict(x0=np.zeros(2),theta=np.array([.8]),variable=np.zeros(1),
        constraint_lower=np.zeros(1),constraint_upper=np.zeros(1),inner_solver_options=io,alm_options=ao)
    expected=None
    for _ in range(3):
        result=solver.solve_lapanda(**kwargs,backward_options=options('subproblem'))
        if expected is None:expected=result
        for key in ('solution','multipliers','penalties','grad_theta'):
            np.testing.assert_array_equal(result[key],expected[key])
        deferred=solver.backward_lapanda_at_solution(result['solution'],kwargs['theta'],kwargs['variable'],
            kwargs['constraint_lower'],kwargs['constraint_upper'],result['multipliers'],result['penalties'],
            result['final_gamma'],backward_options=options('subproblem'))
        np.testing.assert_array_equal(deferred['grad_theta'],result['grad_theta'])
        assert deferred['backward_iterations']==result['backward_iterations']


def test_nonlinear_parameterized_active_constraint(tmp_path):
    x=ca.SX.sym('x',2);theta=ca.SX.sym('theta',3);target=ca.SX.sym('target',2)
    problem=CasadiProblem(u=x,theta=theta,variable=target,cost=.5*ca.sumsqr(x-theta[:2]),
        constraints=ca.vertcat(ca.sumsqr(x)-theta[2]**2),outer_loss=.5*ca.sumsqr(x-target))
    solver=build_solver(problem,backend='compiled',cache_dir=tmp_path,name='native_circle')
    a=np.array([2.,1.]);r=1.;unit=a/np.linalg.norm(a);point=r*unit;z=np.array([.1,-.2])
    lam=(np.linalg.norm(a)/r-1)/2
    result=solver.backward_lapanda_at_solution(point,np.r_[a,r],z,np.array([-1e20]),np.zeros(1),
        np.array([lam]),np.array([10.]),.1,backward_options=options())
    tangent=r/np.linalg.norm(a)*(np.eye(2)-np.outer(unit,unit))
    analytic=np.r_[tangent.T@(point-z),unit@(point-z)]
    np.testing.assert_allclose(result['grad_theta'],analytic,atol=3e-10)
    assert result['backward_active_constraints']==1


def test_original_kkt_rejects_penalty_scaling(tmp_path):
    solver=equality_problem(tmp_path)
    with pytest.raises(ValueError,match='unscaled forward penalties'):
        backward(solver,[.64,.16],-.64,10.,options(scale=100.))


def test_original_kkt_after_one_outer_iteration(tmp_path):
    solver=equality_problem(tmp_path)
    io=SolverOptions();io.tolerance=1e-10;io.max_iterations=2000
    ao=AlmOptions();ao.tolerance=1e-8;ao.max_iterations=30;ao.initial_penalty=10.
    result=solver.solve_lapanda(np.array([.64,.16]),np.array([.8]),np.zeros(1),
        np.zeros(1),np.zeros(1),inner_solver_options=io,alm_options=ao,
        multiplier0=np.array([-.64]),backward_options=options())
    assert result['iterations']==1
    np.testing.assert_allclose(result['grad_theta'],[.544],atol=2e-8)


def test_subproblem_projection_cache_is_per_backward_call(tmp_path):
    solver=equality_problem(tmp_path)
    for theta,rho in [(.8,10.),(.4,2.),(1.,1000.),(.8,10.)]:
        result=backward(solver,[.8*theta,.2*theta],-.8*theta,rho,options('subproblem'),theta)
        np.testing.assert_allclose(result['grad_theta'],[theta*.68*rho/(rho+.8)],atol=2e-10)
        assert result['subproblem_projection_evaluations']==1
        assert result['subproblem_projection_workspace_bytes']>0


def test_subproblem_fixed_box_face_is_gamma_independent(tmp_path):
    solver=equality_problem(tmp_path,True)
    for gamma in [1e-6,.1,1.]:
        result=solver.backward_lapanda_at_solution(np.array([.3,.5]),np.array([.8]),np.zeros(1),
            np.zeros(1),np.zeros(1),np.array([-2.]),np.array([10.]),gamma,
            backward_options=options('subproblem'))
        np.testing.assert_allclose(result['grad_theta'],[5./14.],atol=2e-10)
        assert result['subproblem_projection_evaluations']==1


def test_subproblem_scratch_supports_more_parameters_than_variables(tmp_path):
    x=ca.SX.sym('x',2);theta=ca.SX.sym('theta',3);target=ca.SX.sym('target',2)
    problem=CasadiProblem(u=x,theta=theta,variable=target,cost=.5*ca.sumsqr(x-theta[:2]),
        constraints=ca.vertcat(ca.sumsqr(x)-theta[2]**2),outer_loss=.5*ca.sumsqr(x-target))
    solver=build_solver(problem,backend='compiled',cache_dir=tmp_path,name='reuse_circle')
    a=np.array([2.,1.]);r=1.;unit=a/np.linalg.norm(a);point=r*unit;z=np.array([.1,-.2]);rho=7.
    lam=(np.linalg.norm(a)/r-1)/2;h=np.linalg.norm(a)/r;q=point-z
    v=(q-unit*(unit@q))/h+unit*(unit@q)/(h+4*rho*r*r)
    expected=np.r_[v,4*rho*r*(point@v)]
    result=solver.backward_lapanda_at_solution(point,np.r_[a,r],z,np.array([-1e20]),np.zeros(1),
        np.array([lam]),np.array([rho]),.1,backward_options=options('subproblem'))
    np.testing.assert_allclose(result['grad_theta'],expected,atol=2e-10)
    assert result['subproblem_projection_evaluations']==1
