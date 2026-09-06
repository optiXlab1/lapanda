# Paper Experiments

This directory contains the reproducible experiments used in the lapanda
paper.  Each experiment group has its own direct run and plotting scripts so
that paper text, parameters, and result files can be checked independently.

## Experiment Groups

`exp1_rosenbrock_smooth_constraints`

Nonconvex Rosenbrock-chain benchmark with smooth nonlinear constraints.  The
paper default uses a warm-up size of `N=20`, formal records starting from
`N=50`, and shared solver tolerance `1e-3`.

`exp2_OCPs`

Constrained OCP imitation learning for CartPole, Quadrotor, and Robot Arm.
The paper defaults are defined in `exp2_OCPs/config.py`:

- open-loop: `800` epochs, learning rate `2e-3`
- closed-loop: `500` epochs, learning rate `1e-3`
- horizon `N=20`
- PANDA/ALM tolerances `1e-3/1e-3`
- PANDA/ALM max iterations `1500/20`
- initial ALM penalties: CartPole `10`, Quadrotor `10`, Robot Arm `1e3`
- reported SafePDP baseline: COC mode

Useful direct commands:

```powershell
python experiments\exp2_OCPs\run_open_loop.py
python experiments\exp2_OCPs\run_closed_loop.py
python experiments\exp2_OCPs\plot_open_loop.py
python experiments\exp2_OCPs\plot_closed_loop.py
python experiments\exp2_OCPs\plot_first_mpc_timing.py
```

`exp3_embedded_export`

Embedded obstacle-avoidance experiments for circle and rectangle tasks.  The
paper figures and table data for the rectangle task use the exported C result
in `results/rectangle/c_export_wider_sides_e150/rectangle_margin_imitation_c.csv`
and `results/rectangle/c_export_wider_sides_e150/rectangle_memory.log`.  This is the embedded implementation result rather than the
Python wrapper timing.  The script defaults are:

- Circle lapanda: `N=12`, speed limit `1.5`, PANDA/ALM max iterations
  `2000/100`, tolerances `1e-1/1e-4`, initial penalty `1e4`, penalty update
  factor `10`
- Circle acados: exact mode, SQP/HPIPM, NLP/QP max iterations `1000/200`,
  tolerance `1e-4`
- Rectangle lapanda C export / paper figure: `150` epochs, initial margins
  `(0.15, 0.15, 0.16, 0.22)`, teacher margins
  `(0.12, 0.12, 0.01, 0.22)`, PANDA/ALM max iterations `2000/100`,
  tolerances `1e-3/1e-5`
- Rectangle acados: same rectangle protocol, SQP/HPIPM with tolerance `1e-5`

Useful direct commands:

```powershell
python experiments\exp3_embedded_export\circle\run_lapanda.py --compute-backward
python experiments\exp3_embedded_export\circle\acados.py
python experiments\exp3_embedded_export\rectangle\run_lapanda.py
python experiments\exp3_embedded_export\rectangle\acados.py --backend acados
python experiments\exp3_embedded_export\rectangle\run_smoothed_mpcc_lapanda_rollout.py
python experiments\exp3_embedded_export\rectangle\run_smoothed_mpcc_acados_rollout.py
python experiments\exp3_embedded_export\rectangle\plot_smoothed_mpcc_appendix.py
```
