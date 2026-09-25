# Exp. 2: constrained OCP imitation learning

The three problems are `cartpole`, `quadrotor` and `robot_arm`. Run all
commands from the repository root.

## Paper map

| Paper item | Run | Result used by the paper |
| --- | --- | --- |
| Fig. 3: open-loop loss and timing | `run_open_loop.py`, then `plot_open_loop.py` | `summary_results/ocp_fixed32_2x3_loss_timing.pdf` |
| Fig. 4: closed-loop loss and constraints | `run_closed_loop.py`, then `plot_closed_loop.py` | `summary_results/paper_ocp_1x6_loss_constraints.pdf` |
| Table 3: closed-loop time and memory | `run_closed_loop.py`, then `plot_closed_loop.py` | `summary_results/paper_safepdp_vs_alm_table.csv` |
| Fig. 7: SafePDP barrier comparison | `plot_closed_loop.py` | `summary_results/paper_safepdp_loss_comparison.pdf` |
| Fig. 8: time at each MPC instant | `plot_closed_loop.py` | `summary_results/paper_safepdp_timing_comparison.pdf` |
| Fig. 9: horizon scaling | `run_nonlinear_horizon_scaling.py`, then `plot_horizon_scaling.py` | `summary_results/nonlinear_horizon_scaling_lapanda_turbompc_gpu.pdf` |
| Tables 10 and 11: training and solver settings | no separate run | `config.py` and the JSON files beside each result |
| Table 12: sensitivity accuracy during learning | `run_gradient_checkpoint_diagnostics.py` | `summary_results/ocp_gradient_checkpoint_summary.csv` |

The generated PDFs for Figs. 3, 4, 7 and 8 are copied to the paper as
`figures/fig3.pdf`, `figures/fig4.pdf`, `figures/fig7.pdf` and
`figures/fig8.pdf`. The Fig. 9 file is copied as `figures/horizon_scaling.pdf`.

## Main runs

Run the open-loop and closed-loop experiments:

```powershell
python experiments\exp2_OCPs\run_open_loop.py
python experiments\exp2_OCPs\run_closed_loop.py
```

To run only lapanda:

```powershell
python experiments\exp2_OCPs\run_open_loop.py --methods lapanda
python experiments\exp2_OCPs\run_closed_loop.py --methods lapanda
```

The Robot Arm open-loop run uses the `feasible_v2` teacher archive. The run and
plot scripts select it automatically.

Generate Figs. 3, 4, 7 and 8 and the CSV used for Table 3:

```powershell
python experiments\exp2_OCPs\plot_open_loop.py
python experiments\exp2_OCPs\plot_closed_loop.py
```

The plotting scripts combine the lapanda and SafePDP runs with the TurboMPC
records already stored under `quadrotor/turbompc/results/` and
`robot_arm/turbompc/results/`.

## Appendix results

Table 12 uses four open-loop and four closed-loop checkpoints:

```powershell
python -m experiments.exp2_OCPs.run_gradient_checkpoint_diagnostics
```

Run and plot the Quadrotor horizon test for Fig. 9:

```powershell
python experiments\exp2_OCPs\run_nonlinear_horizon_scaling.py
python experiments\exp2_OCPs\plot_horizon_scaling.py
```

The 32-scenario mean and worst-case statistics are produced by:

```powershell
python -m experiments.exp2_OCPs.run_scenario_statistics
```

The paper runs use penalty scale `1` (the forward-converged penalty) and CG for the
lapanda backward solve, with automatic MINRES fallback. These settings are in
`config.py` and are also saved in the generated JSON files.

`problem_bank.py` defines the OCPs. `common/` contains the shared training and
rollout code, while `safepdp_baseline.py` implements the SafePDP baseline.
