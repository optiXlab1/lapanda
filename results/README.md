# Paper results

`index.csv` links each experiment to its entry point and data. Each learning
method has one retained full run per supported protocol. Per-call records,
saved batches and solver/reference-check flags accompany the summaries.

- `rosenbrock/main`: shared saved forward points, MINRES calibration, held-out
  comparisons, reference checks and backward allocation measurements.
- `rosenbrock/penalty_sweep.csv`: the original fixed-penalty diagnostic, before
  the forward-solver fixes. With the corrected solver and the same 4000-iteration
  budget, the two highest penalties reach the inner limit and produce different
  values; this historical table is not a corrected-version benchmark. The two
  original post-forward strategy tables likewise retain their historical runs.
- `learning`: full loss curves, solve times, settings, checkpoint batches,
  final gradient checks and memory probes for 38 supported runs.
- `learning/illustrative_rollouts`: the original qualitative constraint
  trajectories used in the bottom row of the learning figure.
- `learning/activity`: the separate saved-point backward comparison across
  active-constraint counts; its inputs are independent of the final training runs.
- `sensitivity`: penalty, conditioning and numerical-residual studies.
- `inner_solvers`: three warm/cold PANDA/PANOC repetitions, including failed
  accuracy gates; averages use cases passing both methods' checks.
- `preconditioning`: CG and MINRES experiments at shared inputs.
- `mpc`: the per-step timing records used in the MPC figure. The TurboMPC
  display uses a five-step rolling median; the raw records are retained.
- `embedded`: original device-level summary, retained trajectories and
  smooth-obstacle rollouts. Raw device timing and memory logs are not included.
- `horizon`: discretization study; `status.json` lists completed and pending sizes.

Learning times are cumulative complete solver-call wall times to the first
recorded loss crossing. The full training wall time also includes the outer
learning code. Learning memory is build-plus-first-call incremental RSS;
Rosenbrock and isolated backward studies measure allocated working memory.
These memory metrics have different boundaries.

Gradient errors are percentages unless the file explicitly uses an unscaled
ratio. Batch errors average gradients before taking the relative-error norm.
Reference qualifications and failed stopping checks remain in the records;
they are not filtered out of the retained runs.
