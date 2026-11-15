import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import json, hashlib
from decimal import Decimal, localcontext
import numpy as np
import pandas as pd
from .run_conditioning_sensitivity_v2 import Q, U, N, M
HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parents[2] / 'runs/sensitivity'

def maps(h, sigma, qb, rho):
    diag = h.copy()
    diag[:M] += rho * sigma * sigma
    original = np.r_[Q[:, M:] @ (qb[M:] / h[M:]), U @ (qb[:M] / sigma)]
    proxy = np.r_[Q @ (qb / diag), rho * (U @ (sigma * qb[:M] / diag[:M]))]
    return (original, proxy, diag)

def high_precision(h, sigma, qb, rb, eps, rho):
    with localcontext() as ctx:
        ctx.prec = 80
        D = lambda x: Decimal.from_float(float(x))
        dx = [D(eps) * D(rb[N + i]) / D(sigma[i]) if i < M else D(eps) * D(rb[i]) / D(h[i]) for i in range(N)]
        qh = [D(qb[i]) + dx[i] for i in range(N)]
        diag = [D(h[i]) + (D(rho) * D(sigma[i]) ** 2 if i < M else 0) for i in range(N)]
        v0 = [Decimal(0)] * M + [qh[i] / D(h[i]) for i in range(M, N)]
        v1 = [qh[i] / diag[i] for i in range(N)]
        rotate = lambda matrix, vector: [float(sum((D(matrix[j, i]) * vector[i] for i in range(len(vector))))) for j in range(matrix.shape[0])]
        g0 = np.r_[rotate(Q, v0), rotate(U, [qh[i] / D(sigma[i]) for i in range(M)])]
        g1 = np.r_[rotate(Q, v1), rotate(U, [D(rho) * D(sigma[i]) * v1[i] for i in range(M)])]
        return (g0, g1)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    checks = []
    for delta in [1.0, 0.01, 0.0001, 1e-06]:
        sigma = np.geomspace(1.0, delta, M)
        for mu in [1.0, 0.0001, 1e-08]:
            h = np.r_[np.ones(M), np.geomspace(mu, 1.0, N - M)]
            positive = (h[:M] + np.sqrt(h[:M] ** 2 + 4 * sigma * sigma)) / 2
            negative = 2 * sigma * sigma / (np.sqrt(h[:M] ** 2 + 4 * sigma * sigma) + h[:M])
            condition = max(positive.max(), h[M:].max()) / min(negative.min(), h[M:].min())
            for seed in [11, 23, 41]:
                q = np.random.default_rng(seed).normal(size=N)
                q /= np.linalg.norm(q)
                qb = Q.T @ q
                rb = np.random.default_rng(seed + 1000).normal(size=N + M)
                rb /= np.linalg.norm(rb)
                for rho in [10.0, 1000.0, 100000.0, 10000000.0]:
                    ref, proxy, diag = maps(h, sigma, qb, rho)
                    scale = np.linalg.norm(ref)
                    for eps in [0.0, 1e-09, 1e-06, 0.001]:
                        rhs = eps * rb
                        dx = np.r_[rhs[N:] / sigma, rhs[M:N] / h[M:]]
                        dl = (rhs[:M] - h[:M] * dx[:M]) / sigma
                        measured = np.r_[h * dx + np.r_[sigma * dl, np.zeros(N - M)], sigma * dx[:M]]
                        residual = float(np.linalg.norm(measured))
                        assert np.isclose(residual, eps, rtol=1e-07, atol=1e-20), (delta, mu, eps, residual)
                        original_at_approx, proxy_at_approx, _ = maps(h, sigma, qb + dx, rho)
                        structural = proxy - ref
                        forward = proxy_at_approx - proxy
                        total = proxy_at_approx - ref
                        identity = float(np.linalg.norm(total - structural - forward) / max(np.linalg.norm(total), scale, 1e-300))
                        assert identity < 1e-12
                        rows.append(dict(delta=delta, mu=mu, seed=seed, rho=rho, prescribed_forward_KKT_residual=eps, actual_forward_KKT_residual=residual, original_KKT_condition=condition, subproblem_condition=float(diag.max() / diag.min()), reference_gradient_norm=scale, forward_primal_error=float(np.linalg.norm(dx)), finite_penalty_bias=float(np.linalg.norm(structural) / scale), proxy_forward_only_error=float(np.linalg.norm(forward) / scale), original_forward_only_error=float(np.linalg.norm(original_at_approx - ref) / scale), proxy_total_error=float(np.linalg.norm(total) / scale), decomposition_discrepancy=identity))
                        if seed == 11 and eps == 0.001 and (delta in [1.0, 1e-06]) and (mu in [1.0, 1e-08]) and (rho in [10.0, 10000000.0]):
                            a, b = high_precision(h, sigma, qb, rb, eps, rho)
                            discrepancy = max(np.linalg.norm(a - original_at_approx) / max(np.linalg.norm(a), 1e-300), np.linalg.norm(b - proxy_at_approx) / max(np.linalg.norm(b), 1e-300))
                            checks.append(dict(delta=delta, mu=mu, rho=rho, relative_difference=float(discrepancy)))
                            assert discrepancy < 1e-11, checks[-1]
    data = pd.DataFrame(rows)
    data.to_csv(OUT / 'forward_residual_raw.csv', index=False)
    group = ['delta', 'mu', 'rho', 'prescribed_forward_KKT_residual']
    data.groupby(group).median(numeric_only=True).reset_index().to_csv(OUT / 'forward_residual_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'forward_oracle_checks.csv', index=False)
    (OUT / 'forward_protocol.json').write_text(json.dumps(dict(scope=__doc__, n=N, m=M, original_problem='min .5*x^T H x - theta_f^T x, A x = theta_c; theta=0', outer_loss='q^T x + .5*||x||^2 (same base adjoint RHS q as existing exact-point tests)', residual='Prescribe a common unit direction in orthogonal KKT coordinates, scale by epsilon_F, solve spectrally for (dx,dlambda); verify actual residual.', approximation='Exact reverse at each controlled inexact forward point; no Krylov error.', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitations='Controlled residual-response analysis, not native-ALM stopping, wall-time, or outer-learning experiment.'), indent=2))
    result = dict(complete=True, records=len(data), oracle_checks=len(checks), all_checks_passed=True, max_decomposition_discrepancy=float(data.decomposition_discrepancy.max()))
    (OUT / 'forward_complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
if __name__ == '__main__':
    main()
