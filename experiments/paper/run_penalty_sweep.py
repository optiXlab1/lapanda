from .common import *
from experiments.exp1_rosenbrock_smooth_constraints import run_penalty_gradient_sweep as original
if __name__ == '__main__':
    folder = RUNS / 'penalty_sweep'
    folder.mkdir(exist_ok=True, parents=True)
    original.__file__ = str(folder / 'original_driver.py')
    original.main()
    (folder / 'reference_scope.json').write_text(json.dumps(dict(reference='Same-point active-KKT adjoint at the returned primal-dual point.'), indent=2))
