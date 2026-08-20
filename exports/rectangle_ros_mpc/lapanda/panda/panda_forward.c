#include "panda_forward.h"

#include "panda_eval.h"
#include "lbfgs.h"
#include "matrix_operations.h"

#include <stddef.h>
#include <stdlib.h>
#include <math.h>

/* global solver state */
static panda_params g_params;
static unsigned int g_dimension = 0;
static unsigned char g_initialized = FALSE;
static unsigned char g_has_initial_point = FALSE;

/* accepted previous state and current trial state */
static panda_prev_state g_prev;
static panda_current_state g_current;

/* extra work buffers */
static real_t* g_direction = NULL;       /* current quasi-Newton direction d_k */
static real_t* g_s = NULL;               /* s = x_new - x_old */
static real_t* g_y = NULL;               /* y = res_new - res_old */

/* current accepted tau, kept mainly for logging / inspection */
static real_t g_last_tau = 0.0;

/* gamma-enlargement bookkeeping */
static unsigned int g_stable_gamma_counter = 0;
static real_t g_prev_step_gamma = 0.0;

/* internal helpers */
static int copy_current_to_prev(void);
static void build_current_x_from_prev_and_direction(real_t tau);
static real_t current_normalized_residual_inf_norm(void);
static int try_increase_prev_gamma(real_t* gamma_init);

int panda_forward_init(const panda_params* params)
{
    if (g_initialized == TRUE) {
        panda_forward_cleanup();
    }

    g_params = *params;
    g_dimension = params->n;

    if (panda_alloc_prev_state(&g_prev, g_dimension) == FAILURE) {
        goto fail_1;
    }

    if (panda_alloc_current_state(&g_current, g_dimension) == FAILURE) {
        goto fail_2;
    }

    if (panda_eval_init(g_dimension) == FAILURE) {
        goto fail_3;
    }

    if (lbfgs_init(g_params.lbfgs_memory, g_dimension) == FAILURE) {
        goto fail_4;
    }

    g_direction = malloc(sizeof(real_t) * g_dimension);
    if (g_direction == NULL) {
        goto fail_5;
    }

    g_s = malloc(sizeof(real_t) * g_dimension);
    if (g_s == NULL) {
        goto fail_6;
    }

    g_y = malloc(sizeof(real_t) * g_dimension);
    if (g_y == NULL) {
        goto fail_7;
    }

    g_last_tau = 0.0;
    g_stable_gamma_counter = 0;
    g_prev_step_gamma = 0.0;
    g_initialized = TRUE;
    g_has_initial_point = FALSE;
    return SUCCESS;

fail_7:
    free(g_s);
    g_s = NULL;
fail_6:
    free(g_direction);
    g_direction = NULL;
fail_5:
    lbfgs_cleanup();
fail_4:
    panda_eval_cleanup();
fail_3:
    panda_free_current_state(&g_current);
fail_2:
    panda_free_prev_state(&g_prev);
fail_1:
    return FAILURE;
}

int panda_forward_cleanup(void)
{
    if (g_initialized == FALSE) {
        return SUCCESS;
    }

    free(g_direction);
    free(g_s);
    free(g_y);
    g_direction = NULL;
    g_s = NULL;
    g_y = NULL;

    lbfgs_cleanup();
    panda_eval_cleanup();

    panda_free_current_state(&g_current);
    panda_free_prev_state(&g_prev);

    g_last_tau = 0.0;
    g_stable_gamma_counter = 0;
    g_prev_step_gamma = 0.0;
    g_dimension = 0;
    g_initialized = FALSE;
    g_has_initial_point = FALSE;

    return SUCCESS;
}

int panda_forward_reset(void)
{
    if (g_initialized == FALSE) {
        return FAILURE;
    }

    lbfgs_reset();
    g_last_tau = 0.0;
    g_stable_gamma_counter = 0;
    g_prev_step_gamma = 0.0;
    return SUCCESS;
}

int panda_forward_set_initial(const real_t* x0)
{
    real_t gamma0;

    if (g_initialized == FALSE) {
        return FAILURE;
    }

    gamma0 = panda_eval_initial_gamma(x0, g_params.alpha);
    if (gamma0 < g_params.minimum_gamma) {
        gamma0 = g_params.minimum_gamma;
    }

    if (panda_eval_prev(x0, gamma0, g_params.alpha, &g_prev) == FAILURE) {
        return FAILURE;
    }

    lbfgs_reset();
    g_last_tau = 0.0;
    g_stable_gamma_counter = 0;
    g_prev_step_gamma = g_prev.gamma;
    g_has_initial_point = TRUE;
    return SUCCESS;
}

real_t panda_forward_step(void)
{
    real_t sigma;
    real_t threshold;
    unsigned int tau_backtracks;
    unsigned char can_update_direction;
    real_t gamma_before_step;
    real_t gamma_init;
    real_t norm_residual_map_prev;

    if (g_initialized == FALSE || g_has_initial_point == FALSE) {
        return -1.0;
    }

    /*
     * Try enlarging gamma for the current iteration only.
     * IMPORTANT:
     *   This must NOT overwrite the accepted previous state g_prev.
     */
    gamma_init = g_prev.gamma;
    if (try_increase_prev_gamma(&gamma_init) == FAILURE) {
        return -1.0;
    }

    gamma_before_step = g_prev.gamma;

    /*
     * Step 2.6 right-hand side must use the true accepted previous state:
     *   phi_{k-1} - beta * (1-alpha)/(2 gamma_{k-1}) * ||res_{k-1}||^2
     *
     * So we keep using g_prev.* here, untouched by gamma enlargement.
     */
    sigma = g_params.beta * (0.5 / g_prev.gamma) * (1.0 - g_params.alpha);
    threshold = g_prev.phi - sigma * inner_product(g_prev.res, g_prev.res, g_dimension);

    /*
     * Step 2.1: gamma_k starts from the enlarged candidate (or original gamma).
     * This affects only the current iteration, not g_prev.
     */
    g_current.gamma = gamma_init;

    tau_backtracks = 0;
    can_update_direction = TRUE;

    while (TRUE) {

        if (can_update_direction == TRUE) {
            /*
             * Step 2.2: compute direction d_k and reset tau_k = 1.
             * If the L-BFGS memory is empty, lbfgs_get_direction falls back to -res_prev.
             */
            vector_copy(lbfgs_get_direction(g_prev.res), g_direction, g_dimension);

            g_current.tau = 1.0;
            g_last_tau = g_current.tau;
            tau_backtracks = 0;

            /* Step 2.3 with tau = 1: x_k = x_{k-1} + d_k */
            build_current_x_from_prev_and_direction(g_current.tau);
        } else {
            /*
             * Step 2.3 general form:
             *   x_k = (1-tau_k) z_{k-1} + tau_k (x_{k-1} + d_k)
             */
            build_current_x_from_prev_and_direction(g_current.tau);
            tau_backtracks++;
        }

        /*
         * Step 2.4: evaluate all quantities at the current trial point x_k.
         */
        if (panda_eval_current(
                g_current.x,
                g_current.gamma,
                g_params.alpha,
                &g_current) == FAILURE) {
            return -1.0;
        }

        /*
         * Step 2.5:
         * If upper-model condition fails, shrink gamma and restart direction update.
         */
        if (g_current.upper_ok == FALSE &&
            g_current.gamma > g_params.minimum_gamma) {

            g_current.gamma *= 0.5;
            if (g_current.gamma < g_params.minimum_gamma) {
                g_current.gamma = g_params.minimum_gamma;
            }

            /*
             * Once gamma changes, reset the quasi-Newton memory and
             * recompute direction from scratch.
             */
            lbfgs_reset();
            can_update_direction = TRUE;
            continue;
        }

        /*
         * Step 2.6:
         * accept if FBE decrease is sufficient, or if the maximum number of
         * tau backtracks has been exhausted.
         */
        if (g_current.phi <= threshold ||
            tau_backtracks >= g_params.max_backtracks) {

            if (tau_backtracks >= g_params.max_backtracks) {
                /*
                 * Final fallback: tau = 0 -> current x becomes previous z.
                 * Re-evaluate once more so current state is fully consistent.
                 */
                g_current.tau = 0.0;
                g_last_tau = g_current.tau;

                build_current_x_from_prev_and_direction(g_current.tau);

                if (panda_eval_current(
                        g_current.x,
                        g_current.gamma,
                        g_params.alpha,
                        &g_current) == FAILURE) {
                    return -1.0;
                }
            }

            break;
        }

        /*
         * Otherwise, reduce tau and continue within the same iteration.
         */
        if (tau_backtracks >= g_params.max_backtracks - 1) {
            g_current.tau = 0.0;
        } else {
            g_current.tau *= 0.5;
        }

        g_last_tau = g_current.tau;
        can_update_direction = FALSE;
    }

    /*
     * Accepted update:
     *   s = x_k - x_{k-1}
     *   y = res_k - res_{k-1}
     */
    vector_copy(g_current.x, g_s, g_dimension);
    vector_add_ntimes(g_s, g_prev.x, g_dimension, -1.0);

    vector_copy(g_current.res, g_y, g_dimension);
    vector_add_ntimes(g_y, g_prev.res, g_dimension, -1.0);

    norm_residual_map_prev = vector_norm2(g_prev.res, g_dimension) / g_prev.gamma;
    lbfgs_update(g_s, g_y, norm_residual_map_prev);

    /*
     * Promote current accepted state -> previous state for the next iteration.
     */
    copy_current_to_prev();

    /*
     * Update "stable gamma" counter.
     * Since gamma enlargement no longer overwrites g_prev beforehand,
     * this now correctly reflects accepted gamma changes only.
     */
    if (fabs(g_prev.gamma - gamma_before_step) < MACHINE_ACCURACY) {
        g_stable_gamma_counter++;
    } else {
        g_stable_gamma_counter = 0;
    }
    g_prev_step_gamma = g_prev.gamma;

    return current_normalized_residual_inf_norm();
}

const real_t* panda_forward_get_solution(void)
{
    return g_prev.z;
}

const real_t* panda_forward_get_residual(void)
{
    return g_prev.res;
}

real_t panda_forward_get_gamma(void)
{
    return g_prev.gamma;
}

real_t panda_forward_get_tau(void)
{
    return g_last_tau;
}

real_t panda_forward_get_phi(void)
{
    return g_prev.phi;
}

real_t panda_forward_get_f_x(void)
{
    return g_prev.f_x;
}

real_t panda_forward_get_f_z(void)
{
    return g_prev.f_z;
}

real_t panda_forward_get_g_z(void)
{
    return g_prev.g_z;
}

real_t panda_forward_get_residual_norm2(void)
{
    return vector_norm2(g_prev.res, g_dimension);
}

unsigned char panda_forward_get_upper_ok(void)
{
    return g_prev.upper_ok;
}

int panda_init(const panda_params* params)
{
    return panda_forward_init(params);
}

int panda_cleanup(void)
{
    return panda_forward_cleanup();
}

int panda_reset(void)
{
    return panda_forward_reset();
}

int panda_set_initial(const real_t* x0)
{
    return panda_forward_set_initial(x0);
}

real_t panda_step(void)
{
    return panda_forward_step();
}

const real_t* panda_get_solution(void)
{
    return panda_forward_get_solution();
}

const real_t* panda_get_residual(void)
{
    return panda_forward_get_residual();
}

real_t panda_get_gamma(void)
{
    return panda_forward_get_gamma();
}

real_t panda_get_tau(void)
{
    return panda_forward_get_tau();
}

real_t panda_get_phi(void)
{
    return panda_forward_get_phi();
}

real_t panda_get_f_x(void)
{
    return panda_forward_get_f_x();
}

real_t panda_get_f_z(void)
{
    return panda_forward_get_f_z();
}

real_t panda_get_g_z(void)
{
    return panda_forward_get_g_z();
}

real_t panda_get_residual_norm2(void)
{
    return panda_forward_get_residual_norm2();
}

unsigned char panda_get_upper_ok(void)
{
    return panda_forward_get_upper_ok();
}

/* ===== internal helpers ===== */

static int copy_current_to_prev(void)
{
    vector_copy(g_current.x, g_prev.x, g_dimension);
    g_prev.gamma = g_current.gamma;

    g_prev.f_x = g_current.f_x;
    vector_copy(g_current.df_x, g_prev.df_x, g_dimension);

    vector_copy(g_current.z, g_prev.z, g_dimension);
    g_prev.g_z = g_current.g_z;

    g_prev.f_z = g_current.f_z;
    vector_copy(g_current.df_z, g_prev.df_z, g_dimension);

    vector_copy(g_current.res, g_prev.res, g_dimension);
    g_prev.phi = g_current.phi;
    g_prev.upper_ok = g_current.upper_ok;

    return SUCCESS;
}

static void build_current_x_from_prev_and_direction(real_t tau)
{
    if (tau == 1.0) {
        /*
         * x = x_prev + d
         */
        vector_copy(g_prev.x, g_current.x, g_dimension);
        vector_add_ntimes(g_current.x, g_direction, g_dimension, 1.0);
        return;
    }

    if (tau == 0.0) {
        /*
         * x = z_prev
         */
        vector_copy(g_prev.z, g_current.x, g_dimension);
        return;
    }

    /*
     * x = (1-tau) z_prev + tau (x_prev + d)
     *
     * Start from z_prev, then add:
     *   tau * x_prev + tau * d - tau * z_prev
     */
    vector_copy(g_prev.z, g_current.x, g_dimension);
    vector_add_ntimes(g_current.x, g_prev.x, g_dimension, tau);
    vector_add_ntimes(g_current.x, g_direction, g_dimension, tau);
    vector_add_ntimes(g_current.x, g_prev.z, g_dimension, -tau);
}

static real_t current_normalized_residual_inf_norm(void)
{
    vector_copy(g_prev.res, g_direction, g_dimension);
    vector_real_mul(g_direction, g_dimension, 1.0 / g_prev.gamma);
    return vector_norm_inf(g_direction, g_dimension);
}

static int try_increase_prev_gamma(real_t* gamma_init)
{
    panda_current_state test_state;
    real_t gamma_candidate;
    real_t gamma_last_good;
    unsigned char improved;

    if (g_params.enable_gamma_enlarge == FALSE) {
        return SUCCESS;
    }

    if (g_stable_gamma_counter <= g_params.max_stable_iter) {
        return SUCCESS;
    }

    if (current_normalized_residual_inf_norm() <= g_params.residual_enlarge_threshold) {
        g_stable_gamma_counter = 0;
        return SUCCESS;
    }

    *gamma_init = g_prev.gamma;

    if (panda_alloc_current_state(&test_state, g_dimension) == FAILURE) {
        return FAILURE;
    }

    gamma_last_good = g_prev.gamma;
    gamma_candidate = g_prev.gamma * g_params.gamma_enlarge_factor;
    improved = FALSE;

    while (TRUE) {
        if (panda_eval_current(
                g_prev.x,
                gamma_candidate,
                g_params.alpha,
                &test_state) == FAILURE) {
            break;
        }

        /*
         * Only test local admissibility at x_prev.
         * IMPORTANT:
         *   Do not overwrite g_prev here.
         */
        if (test_state.upper_ok == FALSE || gamma_candidate > 1e2) {
            break;
        }

        gamma_last_good = gamma_candidate;
        improved = TRUE;

        gamma_candidate *= g_params.gamma_enlarge_factor;
    }

    panda_free_current_state(&test_state);

    if (improved == TRUE && gamma_last_good > g_prev.gamma) {
        /*
         * Only pass the enlarged gamma to the current iteration.
         * prev state itself must remain untouched.
         */
        *gamma_init = gamma_last_good;

        /*
         * Since the current-iteration map changed, discard old quasi-Newton memory.
         * But do NOT touch g_prev.{z,res,phi,...}.
         */
        lbfgs_reset();
        g_stable_gamma_counter = 0;
    }

    return SUCCESS;
}
