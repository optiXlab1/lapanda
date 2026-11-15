import argparse
import hashlib
import json
from pathlib import Path

from threadpoolctl import threadpool_limits
from experiments.exp2_OCPs import utils
from .common import RESULTS, RUNS, flat, np, rel
from .references import Reference


def native_coefficients(ins, x, theta, variable, multiplier, penalty):
    c = flat(ins.constraint_fun(x, theta, variable))
    shift = c + multiplier / penalty
    eta = penalty * (shift - np.clip(shift, ins.constraint_lower, ins.constraint_upper))
    active = np.flatnonzero((shift <= ins.constraint_lower + 1e-8) |
        (shift >= ins.constraint_upper - 1e-8) |
        (abs(ins.constraint_upper - ins.constraint_lower) <= 1e-8))
    free = np.flatnonzero((x > ins.problem.box_lower + 1e-10) &
        (x < ins.problem.box_upper - 1e-10))
    return eta, active, free


def main():
    parser = argparse.ArgumentParser(description='Batch-gradient checks against original-NLP and same-point references.')
    parser.add_argument('--model', choices=['cartpole', 'quadrotor', 'robot_arm'], required=True)
    parser.add_argument('--protocol', choices=['open', 'closed'], required=True)
    parser.add_argument('--method', choices=['lapanda-alm', 'lapanda-kkt'], required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    epoch = 800 if args.protocol == 'open' else 500
    source = args.input or RESULTS/'learning'/args.model/args.protocol/args.method/f'epoch{epoch}.npz'
    output = args.output or RUNS/'accuracy'/f'{args.model}_{args.protocol}_{args.method}.json'
    output.parent.mkdir(exist_ok=True, parents=True)
    ins = utils.build_instance(args.model, 20)
    reference = Reference(ins)
    data = np.load(source)
    count = len(data['sample']) if args.samples is None else min(args.samples, len(data['sample']))
    if count <= 0:
        parser.error('--samples must be positive')
    points = []
    with threadpool_limits(1):
        for j in range(count):
            x, theta, variable, gradient = (data[k][j] for k in ['solution', 'theta', 'variable', 'gradient'])
            row = dict(sample=int(data['sample'][j]), gradient=gradient.tolist())
            try:
                gr, sol, info = reference.resolved(x, theta, variable)
                row.update(original_reference=gr.tolist(), reference_qualified=info['qualified'], reference_info=info)
                if j == 0:
                    direction, differences = reference.finite_difference(sol, theta, variable,
                        np.random.default_rng(11).normal(size=len(theta)))
                    row.update(finite_difference_checks=differences,
                        reference_directional_derivative=float(direction @ gr))
            except Exception as exc:
                row.update(original_reference=np.full_like(gradient, np.nan).tolist(),
                    reference_qualified=False, reference_error=repr(exc))
            try:
                eta, active, free = native_coefficients(ins, x, theta, variable,
                    data['multipliers'][j], data['penalties'][j])
                gs, info = reference.gradient(x, theta, variable, eta, active=active, free=free)
                row.update(samepoint_reference=gs.tolist(), samepoint_info=info,
                    samepoint_qualified=bool(info['linear_residual'] <= 1e-8 and
                        info['full_active_adjoint_residual'] <= 1e-8))
            except Exception as exc:
                row.update(samepoint_reference=np.full_like(gradient, np.nan).tolist(),
                    samepoint_qualified=False, samepoint_error=repr(exc))
            points.append(row)
    mean = np.mean([p['gradient'] for p in points], axis=0)
    summary = dict(model=args.model, protocol=args.protocol, method=args.method, samples=count,
        original_batch_error_percent=100*rel(mean, np.mean([p['original_reference'] for p in points], axis=0)),
        samepoint_batch_error_percent=100*rel(mean, np.mean([p['samepoint_reference'] for p in points], axis=0)),
        qualified_original_points=sum(p['reference_qualified'] for p in points),
        qualified_samepoint_points=sum(p['samepoint_qualified'] for p in points),
        snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    output.write_text(json.dumps(dict(summary=summary, points=points), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
