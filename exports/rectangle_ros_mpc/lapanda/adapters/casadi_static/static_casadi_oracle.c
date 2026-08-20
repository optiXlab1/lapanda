#include "static_casadi_oracle.h"

#include "LAPANDA_generated_config.h"

#include <math.h>
#include <string.h>

typedef int (*casadi_eval_fun)(const real_t**, real_t**, long long*, real_t*, int);

static int call_casadi(casadi_eval_fun fun, const real_t** arg, real_t** res)
{
    return fun(arg, res, 0, 0, 0);
}

static real_t static_cost_gradient(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    real_t* output_gradient
)
{
    real_t value;
    const real_t* arg[3];
    real_t* res[2];

    value = 0.0;
    arg[0] = input;
    arg[1] = theta;
    arg[2] = variable;
    res[0] = &value;
    res[1] = output_gradient;
    (void)call_casadi(panda_cost_grad, arg, res);
    return value;
}

static real_t static_prox(real_t* input, real_t gamma, const real_t* theta, const real_t* variable)
{
    unsigned int i;

    (void)gamma;
    (void)theta;
    (void)variable;

    for (i = 0; i < LAPANDA_N; ++i) {
#if LAPANDA_HAS_BOX_LOWER
        if (input[i] < LAPANDA_BOX_LOWER[i]) {
            input[i] = LAPANDA_BOX_LOWER[i];
        }
#endif
#if LAPANDA_HAS_BOX_UPPER
        if (input[i] > LAPANDA_BOX_UPPER[i]) {
            input[i] = LAPANDA_BOX_UPPER[i];
        }
#endif
    }
    return 0.0;
}

static int static_jprox(
    const real_t* u_star,
    real_t gamma,
    const real_t* theta,
    const real_t* variable,
    real_t* J
)
{
    unsigned int i;
    const real_t active_tol = 1e-10;

    (void)gamma;
    (void)theta;
    (void)variable;

    for (i = 0; i < LAPANDA_N; ++i) {
        J[i] = 1.0;
#if LAPANDA_HAS_BOX_LOWER
        if (u_star[i] <= LAPANDA_BOX_LOWER[i] + active_tol) {
            J[i] = 0.0;
        }
#endif
#if LAPANDA_HAS_BOX_UPPER
        if (u_star[i] >= LAPANDA_BOX_UPPER[i] - active_tol) {
            J[i] = 0.0;
        }
#endif
    }
    return SUCCESS;
}

static int static_hvp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* Hv
)
{
    const real_t* arg[4];
    real_t* res[1];

    arg[0] = v;
    arg[1] = u_star;
    arg[2] = theta;
    arg[3] = variable;
    res[0] = Hv;
    return call_casadi(panda_hvp, arg, res) == 0 ? SUCCESS : FAILURE;
}

static int static_loss_grad(
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* L,
    real_t* grad_u_L
)
{
#if LAPANDA_HAS_OUTER_LOSS
    const real_t* arg[3];
    real_t* res[2];

    arg[0] = u_star;
    arg[1] = theta;
    arg[2] = variable;
    res[0] = L;
    res[1] = grad_u_L;
    return call_casadi(panda_loss_grad, arg, res) == 0 ? SUCCESS : FAILURE;
#else
    (void)u_star;
    (void)theta;
    (void)variable;
    *L = 0.0;
    memset(grad_u_L, 0, sizeof(real_t) * LAPANDA_N);
    return SUCCESS;
#endif
}

static int static_vjp(
    const real_t* v,
    const real_t* u_star,
    const real_t* theta,
    const real_t* variable,
    real_t* grad_theta
)
{
    const real_t* arg[4];
    real_t* res[1];

    arg[0] = v;
    arg[1] = u_star;
    arg[2] = theta;
    arg[3] = variable;
    res[0] = grad_theta;
    return call_casadi(panda_vjp, arg, res) == 0 ? SUCCESS : FAILURE;
}

static int static_constraint(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    real_t* constraint
)
{
#if LAPANDA_NCON > 0
    const real_t* arg[3];
    real_t* res[1];

    arg[0] = input;
    arg[1] = theta;
    arg[2] = variable;
    res[0] = constraint;
    return call_casadi(panda_constraints, arg, res) == 0 ? SUCCESS : FAILURE;
#else
    (void)input;
    (void)theta;
    (void)variable;
    (void)constraint;
    return FAILURE;
#endif
}

static int static_constraint_jtprod(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multiplier,
    real_t* output_gradient
)
{
#if LAPANDA_NCON > 0
    const real_t* arg[4];
    real_t* res[1];

    arg[0] = input;
    arg[1] = theta;
    arg[2] = variable;
    arg[3] = multiplier;
    res[0] = output_gradient;
    return call_casadi(panda_constraint_jtprod, arg, res) == 0 ? SUCCESS : FAILURE;
#else
    (void)input;
    (void)theta;
    (void)variable;
    (void)multiplier;
    (void)output_gradient;
    return FAILURE;
#endif
}

static void projection_data(
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multipliers,
    const real_t* penalties,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    real_t* projected_multiplier,
    real_t* active_penalty
)
{
    unsigned int i;
    real_t c_value[LAPANDA_NCON > 0 ? LAPANDA_NCON : 1];

    (void)static_constraint(input, theta, variable, c_value);
    for (i = 0; i < LAPANDA_NCON; ++i) {
        real_t shifted = c_value[i] + multipliers[i] / penalties[i];
        real_t projected = shifted;
        int active;
        int equality;
        if (projected < constraint_lower[i]) {
            projected = constraint_lower[i];
        }
        if (projected > constraint_upper[i]) {
            projected = constraint_upper[i];
        }
        equality = fabs(constraint_upper[i] - constraint_lower[i]) <= 1e-8;
        active =
            shifted <= constraint_lower[i] + 1e-8 ||
            shifted >= constraint_upper[i] - 1e-8 ||
            equality;
        projected_multiplier[i] = penalties[i] * (shifted - projected);
        active_penalty[i] = active ? penalties[i] : 0.0;
    }
}

static int static_constraint_hvp(
    const real_t* v,
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multipliers,
    const real_t* penalties,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    real_t* Hv
)
{
#if LAPANDA_NCON > 0
    unsigned int i;
    real_t projected_multiplier[LAPANDA_NCON];
    real_t active_penalty[LAPANDA_NCON];
    real_t jv[LAPANDA_NCON];
    real_t curvature[LAPANDA_N];
    real_t gn_weight[LAPANDA_NCON];
    real_t gauss_newton[LAPANDA_N];
    const real_t* arg_jv[4];
    const real_t* arg_weighted[5];
    real_t* res[1];

    projection_data(input, theta, variable, multipliers, penalties,
                    constraint_lower, constraint_upper,
                    projected_multiplier, active_penalty);

    arg_jv[0] = v;
    arg_jv[1] = input;
    arg_jv[2] = theta;
    arg_jv[3] = variable;
    res[0] = jv;
    if (call_casadi(panda_constraint_jv, arg_jv, res) != 0) return FAILURE;

    arg_weighted[0] = v;
    arg_weighted[1] = input;
    arg_weighted[2] = theta;
    arg_weighted[3] = variable;
    arg_weighted[4] = projected_multiplier;
    res[0] = curvature;
    if (call_casadi(panda_constraint_weighted_hvp, arg_weighted, res) != 0) return FAILURE;

    for (i = 0; i < LAPANDA_NCON; ++i) {
        gn_weight[i] = active_penalty[i] * jv[i];
    }
    if (static_constraint_jtprod(input, theta, variable, gn_weight, gauss_newton) == FAILURE) {
        return FAILURE;
    }
    for (i = 0; i < LAPANDA_N; ++i) {
        Hv[i] = curvature[i] + gauss_newton[i];
    }
    return SUCCESS;
#else
    (void)v; (void)input; (void)theta; (void)variable; (void)multipliers;
    (void)penalties; (void)constraint_lower; (void)constraint_upper; (void)Hv;
    return FAILURE;
#endif
}

static int static_constraint_vjp(
    const real_t* v,
    const real_t* input,
    const real_t* theta,
    const real_t* variable,
    const real_t* multipliers,
    const real_t* penalties,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    real_t* grad_theta
)
{
#if LAPANDA_NCON > 0
    unsigned int i;
    real_t projected_multiplier[LAPANDA_NCON];
    real_t active_penalty[LAPANDA_NCON];
    real_t jv[LAPANDA_NCON];
    real_t curvature[LAPANDA_NTHETA];
    real_t gn_weight[LAPANDA_NCON];
    real_t gauss_newton[LAPANDA_NTHETA];
    const real_t* arg_jv[4];
    const real_t* arg_weighted[5];
    const real_t* arg_theta[4];
    real_t* res[1];

    projection_data(input, theta, variable, multipliers, penalties,
                    constraint_lower, constraint_upper,
                    projected_multiplier, active_penalty);

    arg_jv[0] = v;
    arg_jv[1] = input;
    arg_jv[2] = theta;
    arg_jv[3] = variable;
    res[0] = jv;
    if (call_casadi(panda_constraint_jv, arg_jv, res) != 0) return FAILURE;

    arg_weighted[0] = v;
    arg_weighted[1] = input;
    arg_weighted[2] = theta;
    arg_weighted[3] = variable;
    arg_weighted[4] = projected_multiplier;
    res[0] = curvature;
    if (call_casadi(panda_constraint_weighted_vjp, arg_weighted, res) != 0) return FAILURE;

    for (i = 0; i < LAPANDA_NCON; ++i) {
        gn_weight[i] = active_penalty[i] * jv[i];
    }
    arg_theta[0] = input;
    arg_theta[1] = theta;
    arg_theta[2] = variable;
    arg_theta[3] = gn_weight;
    res[0] = gauss_newton;
    if (call_casadi(panda_constraint_theta_jtprod, arg_theta, res) != 0) return FAILURE;

    for (i = 0; i < LAPANDA_NTHETA; ++i) {
        grad_theta[i] = curvature[i] + gauss_newton[i];
    }
    return SUCCESS;
#else
    (void)v; (void)input; (void)theta; (void)variable; (void)multipliers;
    (void)penalties; (void)constraint_lower; (void)constraint_upper; (void)grad_theta;
    return FAILURE;
#endif
}

static void fill_oracle(panda_oracle* oracle)
{
    oracle->n = LAPANDA_N;
    oracle->ntheta = LAPANDA_NTHETA;
    oracle->nvar = LAPANDA_NVAR;
    oracle->cost_gradient = static_cost_gradient;
    oracle->proxg = static_prox;
    oracle->jprox = static_jprox;
    oracle->hvp = static_hvp;
    oracle->loss_grad = static_loss_grad;
    oracle->vjp = static_vjp;
}

void LAPANDA_static_init_panda_problem(
    struct optimizer_problem* problem,
    const struct solver_parameters* solver_params,
    const struct backward_parameters* backward_params
)
{
    fill_oracle(&problem->oracle);
    if (solver_params != 0) {
        problem->solver_params = *solver_params;
    }
    if (backward_params != 0) {
        problem->backward_params = *backward_params;
    }
    problem->trace = 0;
    problem->trace_context = 0;
}

void LAPANDA_static_init_alm_problem(
    alm_problem* problem,
    const real_t* constraint_lower,
    const real_t* constraint_upper,
    const struct solver_parameters* inner_solver_params,
    const struct backward_parameters* backward_params
)
{
    fill_oracle(&problem->oracle);
    problem->ncon = LAPANDA_NCON;
    problem->constraint_lower = constraint_lower;
    problem->constraint_upper = constraint_upper;
    problem->constraint = static_constraint;
    problem->constraint_jtprod = static_constraint_jtprod;
    problem->constraint_hvp = static_constraint_hvp;
    problem->constraint_vjp = static_constraint_vjp;
    if (inner_solver_params != 0) {
        problem->inner_solver_params = *inner_solver_params;
    }
    if (backward_params != 0) {
        problem->backward_params = *backward_params;
    }
}
