# Exp.2 OCP Experiments

This directory contains the three constrained OCP imitation-learning examples:

- `cartpole`
- `quadrotor`
- `robot_arm`

The useful entry points are intentionally small:

```text
run_closed_loop.py   closed-loop MPC rollout imitation, including SafePDP-COC
run_open_loop.py     fixed teacher-archive imitation, including SafePDP-COC
plot_closed_loop.py  closed-loop paper figures and summary table
plot_open_loop.py    open-loop paper 2x3 loss/timing figure and summary table
plot_first_mpc_timing.py  first MPC rollout timing figure
run_nonlinear_horizon_scaling.py  run the lapanda side of the nonlinear Quadrotor scaling benchmark
plot_horizon_scaling.py  reproduce the lapanda/TurboMPC-GPU scaling figure
config.py            per-problem defaults, including initial ALM penalty
```

The remaining files are support code:

```text
problem_bank.py      OCP definitions
utils.py             shared lapanda helpers
safepdp_baseline.py  shared SafePDP implementation
common/              implementation blocks used by the two run scripts
```

Typical commands:

```powershell
python experiments\exp2_OCPs\run_closed_loop.py
python experiments\exp2_OCPs\run_open_loop.py
python experiments\exp2_OCPs\plot_closed_loop.py
python experiments\exp2_OCPs\plot_open_loop.py
python experiments\exp2_OCPs\plot_first_mpc_timing.py
python experiments\exp2_OCPs\run_nonlinear_horizon_scaling.py
python experiments\exp2_OCPs\plot_horizon_scaling.py
```

Run only lapanda:

```powershell
python experiments\exp2_OCPs\run_closed_loop.py --methods lapanda
python experiments\exp2_OCPs\run_open_loop.py --methods lapanda
```

Set different initial ALM penalties:

```powershell
python experiments\exp2_OCPs\run_closed_loop.py --cartpole-alm-initial-penalty 100 --quadrotor-alm-initial-penalty 1000 --robot-arm-alm-initial-penalty 1000
```

The paper default SafePDP baseline is COC mode.  Barrier mode is kept only for
diagnostic runs:

```python
SAFEPDP_DEFAULTS = {
    "cartpole": SafePdpDefaults(sensitivity_mode="coc", barrier_gamma=1e-2),
    "quadrotor": SafePdpDefaults(sensitivity_mode="coc", barrier_gamma=1e-2),
    "robot_arm": SafePdpDefaults(sensitivity_mode="coc", barrier_gamma=1e-2),
}
```

Results are written to:

```text
cartpole/results
quadrotor/results
robot_arm/results
summary_results
```

The open-loop paper outputs are:

```text
summary_results/ocp_fixed32_2x3_loss_timing.{png,svg,pdf}
summary_results/open_loop_800_timing_summary.csv
```

The closed-loop plotting command regenerates the timing, loss, and constraint
figures directly from the retained experiment results:

```text
summary_results/paper_safepdp_timing_comparison.{png,svg,pdf}
summary_results/paper_safepdp_loss_comparison.{png,svg,pdf}
summary_results/paper_safepdp_constraint_bands.{png,svg,pdf}
summary_results/paper_ocp_1x6_loss_constraints.{png,svg,pdf}
summary_results/paper_safepdp_vs_alm_table.csv
```

The first-MPC timing figure is:

```text
summary_results/ocp_first_mpc_timing.{png,svg,pdf}
```

The horizon-scaling outputs are:

```text
summary_results/nonlinear_horizon_scaling_lapanda_turbompc_gpu.{png,svg,pdf}
summary_results/nonlinear_horizon_scaling_lapanda_turbompc_gpu.csv
```

The plotting command reads the retained summary CSV directly; exploratory
cold-start, fixed-step, linear, and large-horizon sweeps are not part of the
paper reproduction package.
