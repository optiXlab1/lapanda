#ifndef PANDA_TYPES_H
#define PANDA_TYPES_H

#include "../globals/globals.h"

typedef struct {
    unsigned int n;
    real_t alpha;               /* upper-model parameter, e.g. 0.95 */
    real_t beta;                /* FBE decrease parameter, e.g. 0.5 */
    real_t minimum_gamma;       /* lower bound for adaptive gamma */
    unsigned int max_backtracks;
    unsigned int lbfgs_memory;

    /* optional gamma enlargement controls */
    unsigned char enable_gamma_enlarge;
    unsigned int max_stable_iter;
    real_t residual_enlarge_threshold;
    real_t gamma_enlarge_factor;
} panda_params;

/*
 * Previous accepted state: quantities associated with x_{k-1}.
 * These remain fixed during the inner current-state trial loop.
 */
typedef struct {
    real_t* x;                  /* previous accepted iterate x_{k-1} */
    real_t gamma;               /* previous accepted gamma_{k-1} */

    real_t f_x;                 /* f(x_{k-1}) */
    real_t* df_x;               /* grad f(x_{k-1}) */

    real_t* z;                  /* previous forward-backward point z_{k-1} */
    real_t g_z;                 /* g(z_{k-1}) */

    real_t f_z;                 /* f(z_{k-1}) */
    real_t* df_z;               /* grad f(z_{k-1}) */

    real_t* res;                /* residual gap: x_{k-1} - z_{k-1} */
    real_t phi;                 /* FBE at (x_{k-1}, gamma_{k-1}) */
    unsigned char upper_ok;     /* whether upper-model condition holds */
} panda_prev_state;

/*
 * Current trial state: quantities associated with the current x_k
 */
typedef struct {
    real_t* x;                  /* current trial iterate x_k */
    real_t gamma;               /* current gamma_k */
    real_t tau;                 /* current tau_k */

    real_t f_x;                 /* f(x_k) */
    real_t* df_x;               /* grad f(x_k) */

    real_t* z;                  /* current forward-backward point z_k */
    real_t g_z;                 /* g(z_k) */

    real_t f_z;                 /* f(z_k) */
    real_t* df_z;               /* grad f(z_k) */

    real_t* res;                /* residual gap: x_k - z_k */
    real_t phi;                 /* FBE at (x_k, gamma_k) */

    unsigned char upper_ok;     /* whether upper-model condition holds */
} panda_current_state;

#endif