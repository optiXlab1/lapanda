#include "panda_eval.h"

#include "function_evaluator.h"
#include "matrix_operations.h"

#include <stdlib.h>
#include <math.h>

/* internal work buffers */
static unsigned int g_dimension = 0;
static real_t* g_tmp_x = NULL;      /* generic temporary vector */
static real_t* g_tmp_df = NULL;     /* gradient buffer for shifted point */

/*
 * Build a small perturbation x_shift around x and return ||x_shift - x||_2.
 * This mirrors the spirit of the old lipschitz.c implementation, but without
 * relying on buffer.*. The evaluator is now self-contained.
 */
static real_t build_shift_and_get_norm(const real_t* x, real_t* x_shift)
{
    real_t norm_sq = 0.0;
    unsigned int i;

    for (i = 0; i < g_dimension; ++i) {
        real_t delta;

        if (fabs(x[i]) < DELTA_LIPSCHITZ_SAFETY_VALUE) {
            delta = DELTA_LIPSCHITZ;
        } else {
            delta = fabs(x[i]) * DELTA_LIPSCHITZ_SAFETY_VALUE;
        }

        x_shift[i] = x[i] + delta;
        norm_sq += delta * delta;
    }

    return sqrt(norm_sq);
}

/*
 * Estimate a local Lipschitz constant by finite differencing the gradient:
 * L ≈ ||grad f(x + delta) - grad f(x)|| / ||delta||.
 */
static real_t estimate_local_lipschitz(const real_t* x, const real_t* df_x)
{
    real_t denominator;
    real_t numerator;

    denominator = build_shift_and_get_norm(x, g_tmp_x);
    function_evaluator_f_df(g_tmp_x, g_tmp_df);

    /* reuse g_tmp_x as a difference buffer: g_tmp_x = g_tmp_df - df_x */
    vector_copy(g_tmp_df, g_tmp_x, g_dimension);
    vector_add_ntimes(g_tmp_x, df_x, g_dimension, -1.0);

    numerator = vector_norm2(g_tmp_x, g_dimension);

    return numerator / denominator;
}

/*
 * Compute FBE_gamma(x) = f(x) + g(z) - <grad f(x), x-z> + ||x-z||^2/(2 gamma)
 * where z = prox(x - gamma grad f(x)).
 */
static real_t compute_fbe(
    real_t f_x,
    const real_t* df_x,
    real_t g_z,
    const real_t* res,
    real_t gamma
)
{
    real_t res_norm_sq = inner_product(res, res, g_dimension);

    return f_x
        + g_z
        - inner_product(df_x, res, g_dimension)
        + 0.5 * res_norm_sq / gamma;
}

/*
 * Check the local upper-model condition:
 * f(z) <= f(x) + <grad f(x), z-x> + alpha/(2 gamma) ||z-x||^2
 *
 * Since res = x-z, this becomes:
 * f(z) <= f(x) - <grad f(x), res> + alpha/(2 gamma) ||res||^2.
 */
static unsigned char check_upper_model(
    real_t f_x,
    const real_t* df_x,
    real_t f_z,
    const real_t* res,
    real_t alpha,
    real_t gamma
)
{
    real_t rhs;

    rhs = f_x
        - inner_product(df_x, res, g_dimension)
        + (alpha / (2.0 * gamma)) * inner_product(res, res, g_dimension);

    if (f_z <= rhs + 1e-10 * (1.0 + fabs(f_x))) {
        return TRUE;
    }
    return FALSE;
}

/*
 * Common internal evaluation routine used for both previous and current states.
 * It computes:
 *   f_x, df_x,
 *   z = prox(x - gamma df_x), g_z,
 *   f_z, df_z,
 *   res = x-z,
 *   phi = FBE,
 *   upper_ok.
 */
static int evaluate_common(
    const real_t* x,
    real_t gamma,
    real_t alpha,
    real_t* f_x,
    real_t* df_x,
    real_t* z,
    real_t* g_z,
    real_t* f_z,
    real_t* df_z,
    real_t* res,
    real_t* phi,
    unsigned char* upper_ok
)
{
    /* Step 1: evaluate f(x), grad f(x) */
    *f_x = function_evaluator_f_df(x, df_x);

    /* Step 2: form v = x - gamma * grad f(x) into g_tmp_x */
    vector_copy(x, g_tmp_x, g_dimension);
    vector_add_ntimes(g_tmp_x, df_x, g_dimension, -gamma);

    /*
     * Step 3: proxg writes the proximal point into g_tmp_x and returns g(z).
     * This matches the existing function_evaluator_proxg interface.
     */
    *g_z = function_evaluator_proxg(g_tmp_x, gamma);

    /* store z */
    vector_copy(g_tmp_x, z, g_dimension);

    /* Step 4: evaluate f(z), grad f(z) */
    *f_z = function_evaluator_f_df(z, df_z);

    /* Step 5: residual gap res = x - z */
    vector_copy(x, res, g_dimension);
    vector_add_ntimes(res, z, g_dimension, -1.0);

    /* Step 6: FBE */
    *phi = compute_fbe(*f_x, df_x, *g_z, res, gamma);

    /* Step 7: upper-model condition */
    *upper_ok = check_upper_model(*f_x, df_x, *f_z, res, alpha, gamma);

    return SUCCESS;
}

int panda_eval_init(unsigned int dimension_problem)
{
    g_dimension = dimension_problem;

    g_tmp_x = malloc(sizeof(real_t) * g_dimension);
    if (g_tmp_x == NULL) {
        return FAILURE;
    }

    g_tmp_df = malloc(sizeof(real_t) * g_dimension);
    if (g_tmp_df == NULL) {
        free(g_tmp_x);
        g_tmp_x = NULL;
        return FAILURE;
    }

    return SUCCESS;
}

int panda_eval_cleanup(void)
{
    free(g_tmp_x);
    free(g_tmp_df);

    g_tmp_x = NULL;
    g_tmp_df = NULL;
    g_dimension = 0;

    return SUCCESS;
}

real_t panda_eval_initial_gamma(const real_t* x, real_t alpha)
{
    real_t L;
    real_t f_dummy;

    /* reuse g_tmp_df as grad buffer at x */
    f_dummy = function_evaluator_f_df(x, g_tmp_df);
    (void)f_dummy;

    L = estimate_local_lipschitz(x, g_tmp_df);
    if (L < 1e-2) {
        L = 1e-2;
    }

    return alpha / L;
}

int panda_alloc_prev_state(panda_prev_state* state, unsigned int n)
{
    state->x = malloc(sizeof(real_t) * n);
    state->df_x = malloc(sizeof(real_t) * n);
    state->z = malloc(sizeof(real_t) * n);
    state->df_z = malloc(sizeof(real_t) * n);
    state->res = malloc(sizeof(real_t) * n);

    if (state->x == NULL ||
        state->df_x == NULL ||
        state->z == NULL ||
        state->df_z == NULL ||
        state->res == NULL) {
        return FAILURE;
    }

    state->gamma = 0.0;
    state->f_x = 0.0;
    state->g_z = 0.0;
    state->f_z = 0.0;
    state->phi = 0.0;

    return SUCCESS;
}

int panda_free_prev_state(panda_prev_state* state)
{
    free(state->x);
    free(state->df_x);
    free(state->z);
    free(state->df_z);
    free(state->res);

    state->x = NULL;
    state->df_x = NULL;
    state->z = NULL;
    state->df_z = NULL;
    state->res = NULL;

    return SUCCESS;
}

int panda_alloc_current_state(panda_current_state* state, unsigned int n)
{
    state->x = malloc(sizeof(real_t) * n);
    state->df_x = malloc(sizeof(real_t) * n);
    state->z = malloc(sizeof(real_t) * n);
    state->df_z = malloc(sizeof(real_t) * n);
    state->res = malloc(sizeof(real_t) * n);

    if (state->x == NULL ||
        state->df_x == NULL ||
        state->z == NULL ||
        state->df_z == NULL ||
        state->res == NULL) {
        return FAILURE;
    }

    state->gamma = 0.0;
    state->tau = 0.0;
    state->f_x = 0.0;
    state->g_z = 0.0;
    state->f_z = 0.0;
    state->phi = 0.0;
    state->upper_ok = FALSE;

    return SUCCESS;
}

int panda_free_current_state(panda_current_state* state)
{
    free(state->x);
    free(state->df_x);
    free(state->z);
    free(state->df_z);
    free(state->res);

    state->x = NULL;
    state->df_x = NULL;
    state->z = NULL;
    state->df_z = NULL;
    state->res = NULL;

    return SUCCESS;
}

int panda_eval_prev(
    const real_t* x,
    real_t gamma,
    real_t alpha,
    panda_prev_state* out
)
{
    unsigned char upper_dummy;

    vector_copy(x, out->x, g_dimension);
    out->gamma = gamma;

    return evaluate_common(
        out->x,
        gamma,
        alpha,
        &out->f_x,
        out->df_x,
        out->z,
        &out->g_z,
        &out->f_z,
        out->df_z,
        out->res,
        &out->phi,
        &upper_dummy
    );
}

int panda_eval_current(
    const real_t* x,
    real_t gamma,
    real_t alpha,
    panda_current_state* out
)
{
    vector_copy(x, out->x, g_dimension);
    out->gamma = gamma;

    return evaluate_common(
        out->x,
        gamma,
        alpha,
        &out->f_x,
        out->df_x,
        out->z,
        &out->g_z,
        &out->f_z,
        out->df_z,
        out->res,
        &out->phi,
        &out->upper_ok
    );
}
