import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, {4})
import json, time, hashlib, argparse
from pathlib import Path
from decimal import Decimal, localcontext
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.sparse.linalg import LinearOperator, cg, minres
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parents[2] / 'runs/sensitivity'
N, M = (64, 16)
Q = hadamard(N).astype(float) / 8
U = hadamard(M).astype(float) / 4
assert np.array_equal(Q.T @ Q, np.eye(N)) and np.array_equal(U.T @ U, np.eye(M))

def norm_error(g, ref):
    return float(np.linalg.norm(g - ref) / np.linalg.norm(ref))

def oracle(h, sigma, q, rho):
    qb = Q.T @ q
    vref = Q[:, M:] @ (qb[M:] / h[M:])
    wref = U @ (qb[:M] / sigma)
    reference = np.r_[vref, wref]
    diagonal = h.copy()
    diagonal[:M] += rho * sigma ** 2
    v = Q @ (qb / diagonal)
    frozen = np.r_[v, rho * (U @ (sigma * (qb[:M] / diagonal[:M])))]
    return (reference, frozen, diagonal)

def decimal_oracle(h, sigma, q, rho):
    with localcontext() as ctx:
        ctx.prec = 80
        D = lambda x: Decimal.from_float(float(x))
        qb = [sum((D(Q[j, i]) * D(q[j]) for j in range(N))) for i in range(N)]
        tangent = [Decimal(0)] * M + [qb[i] / D(h[i]) for i in range(M, N)]
        vref = [sum((D(Q[j, i]) * tangent[i] for i in range(N))) for j in range(N)]
        wref = [sum((D(U[j, i]) * qb[i] / D(sigma[i]) for i in range(M))) for j in range(M)]
        diag = [D(h[i]) + (D(rho) * D(sigma[i]) ** 2 if i < M else 0) for i in range(N)]
        v = [sum((D(Q[j, i]) * qb[i] / diag[i] for i in range(N))) for j in range(N)]
        w = [D(rho) * sum((D(U[j, i]) * D(sigma[i]) * qb[i] / diag[i] for i in range(M))) for j in range(M)]
        return (np.array([float(x) for x in vref + wref]), np.array([float(x) for x in v + w]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stopping', choices=['common_residual', 'native'], default='common_residual')
    args = parser.parse_args()
    prefix = 'common_residual_' if args.stopping == 'common_residual' else ''
    OUT.mkdir(parents=True, exist_ok=True)
    raw = []
    exact = []
    validation = []
    config = dict(scope=__doc__, n=N, m=M, seeds=[11, 23, 41], minimum_constraint_singular_values=[1.0, 0.01, 0.0001, 1e-06], minimum_tangent_curvatures=[1.0, 0.0001, 1e-08], rho=[10.0, 1000.0, 100000.0, 10000000.0], linear_tolerances=[0.001, 1e-06, 1e-09], max_iterations=4000, reference='Analytic spectral formulas; selected moderate/extreme cases cross-checked with Decimal80.', numerical_implementation='SciPy matrix-free CG/MINRES, double, single thread, CPU4. Diagnostic only; no timing comparison with native-C primary experiments.', criterion='Both methods use a callback to stop at the same actual unscaled Euclidean relative residual; native rtol=0. Internal machine stops/caps before this gate remain explicitly unqualified.' if prefix else 'Native stopping. Record native status and actual Euclidean residual separately. Same numeric tolerance is not asserted to be same attained accuracy.', parameterization='Unscaled independent objective-linear and constraint-RHS parameters. Near-dependent constraints change the parameter solution response, rather than jointly rescaling A and its parameter RHS.', metrics='Total error to original gradient, bias from analytic frozen gradient, numerical solve error to each method target, constraint-RHS gradient block error.', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / (prefix + 'protocol.json')).write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    with threadpool_limits(1):
        for delta in config['minimum_constraint_singular_values']:
            sigma = np.geomspace(1.0, delta, M)
            for mu in config['minimum_tangent_curvatures']:
                h = np.r_[np.ones(M), np.geomspace(mu, 1.0, N - M)]
                positive = (h[:M] + np.sqrt(h[:M] ** 2 + 4 * sigma ** 2)) / 2
                negative_abs = 2 * sigma ** 2 / (np.sqrt(h[:M] ** 2 + 4 * sigma ** 2) + h[:M])
                kmin = min(negative_abs.min(), h[M:].min())
                kmax = max(positive.max(), h[M:].max())
                for seed in config['seeds']:
                    q = np.random.default_rng(seed).normal(size=N)
                    q /= np.linalg.norm(q)
                    for rho in config['rho']:
                        ref, frozen, diagonal = oracle(h, sigma, q, rho)
                        if seed == 11 and rho in [10.0, 10000000.0] and ((delta, mu) in [(1.0, 1.0), (1e-06, 1.0), (1.0, 1e-08), (1e-06, 1e-08)]):
                            high_ref, high_frozen = decimal_oracle(h, sigma, q, rho)
                            validation.append(dict(delta=delta, mu=mu, rho=rho, original_oracle_relative_difference=norm_error(ref, high_ref), frozen_oracle_relative_difference=norm_error(frozen, high_frozen)))
                        vb = Q.T @ q / diagonal
                        identity = np.r_[Q[:, :M] @ vb[:M], -U @ (h[:M] / sigma * vb[:M])]
                        discrepancy = np.linalg.norm(frozen - ref - identity) / max(np.linalg.norm(ref), np.linalg.norm(frozen))
                        assert discrepancy < 1e-11, (delta, mu, rho, discrepancy)
                        base = dict(delta=delta, mu=mu, seed=seed, rho=rho, kappa_A=1 / delta, kappa_K=kmax / kmin, K_inverse_norm=1 / kmin, kappa_M=float(diagonal.max() / diagonal.min()), effective_weak_penalty=rho * delta ** 2, exact_gradient_norm=float(np.linalg.norm(ref)), finite_penalty_error=norm_error(frozen, ref), constraint_block_bias=norm_error(frozen[N:], ref[N:]), bias_identity_discrepancy=discrepancy)
                        exact.append(base)

                        def Hmul(x):
                            return Q @ (h * (Q.T @ x))

                        def Amul(x):
                            return U @ (sigma * (Q[:, :M].T @ x))

                        def ATmul(y):
                            return Q[:, :M] @ (sigma * (U.T @ y))

                        def Kmul(z):
                            return np.r_[Hmul(z[:N]) + ATmul(z[N:]), Amul(z[:N])]

                        def Mmul(x):
                            return Q @ (diagonal * (Q.T @ x))
                        for tolerance in config['linear_tolerances']:
                            for method in ['cg', 'minres']:
                                count = [0]
                                operator = Mmul if method == 'cg' else Kmul
                                rhs = q if method == 'cg' else np.r_[q, np.zeros(M)]
                                rhsnorm = np.linalg.norm(rhs)

                                class ResidualGate(Exception):
                                    pass
                                stopped = [None]

                                def callback(x):
                                    count[0] += 1
                                    if prefix and np.linalg.norm(operator(x) - rhs) <= tolerance * rhsnorm:
                                        stopped[0] = x.copy()
                                        raise ResidualGate()
                                try:
                                    try:
                                        if method == 'cg':
                                            solution, native_status = cg(LinearOperator((N, N), matvec=Mmul, dtype=float), q, rtol=0 if prefix else tolerance, atol=0, maxiter=4000, callback=callback)
                                        else:
                                            solution, native_status = minres(LinearOperator((N + M, N + M), matvec=Kmul, dtype=float), rhs, rtol=0 if prefix else tolerance, maxiter=4000, callback=callback, check=False)
                                    except ResidualGate:
                                        solution = stopped[0]
                                        native_status = 0
                                    if method == 'cg':
                                        actual = float(np.linalg.norm(Mmul(solution) - q) / np.linalg.norm(q))
                                        g = np.r_[solution, rho * Amul(solution)]
                                        target = frozen
                                    else:
                                        actual = float(np.linalg.norm(Kmul(solution) - rhs) / np.linalg.norm(rhs))
                                        g = solution
                                        target = ref
                                    status = 0 if actual <= 1.05 * tolerance else int(native_status) if native_status else -2
                                    raw.append(dict(**base, method=method, tolerance=tolerance, status=int(status), native_status=int(native_status), iterations=count[0], actual_relative_residual=actual, actual_residual_gate=actual <= 1.05 * tolerance, gradient_error=norm_error(g, ref), linear_only_gradient_error=norm_error(g, target), absolute_gradient_error=float(np.linalg.norm(g - ref)), constraint_block_error=norm_error(g[N:], ref[N:]), error=''))
                                except Exception as error:
                                    raw.append(dict(**base, method=method, tolerance=tolerance, status=-999, iterations=count[0], actual_residual_gate=False, error=repr(error)))
                print('DONE', delta, mu, 'records', len(raw), flush=True)
                pd.DataFrame(raw).to_csv(OUT / (prefix + 'raw.csv'), index=False)
    pd.DataFrame(exact).to_csv(OUT / (prefix + 'analytic.csv'), index=False)
    pd.DataFrame(validation).to_csv(OUT / (prefix + 'oracle_validation.csv'), index=False)
    assert len(raw) == 864 and len(exact) == 144
    assert max((max(r['original_oracle_relative_difference'], r['frozen_oracle_relative_difference']) for r in validation)) < 1e-12
    result = dict(complete=True, records=len(raw), analytic_points=len(exact), oracle_checks=len(validation), oracle_checks_passed=True, bias_identity_checks_passed=True, elapsed_sec=time.monotonic() - started, native_solver_status_counts=pd.DataFrame(raw).groupby(['method', 'status']).size().to_dict().__str__(), scope='Accuracy and conditioning diagnosis only. Failed/insufficient actual residual gates are retained.')
    (OUT / (prefix + 'complete.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
