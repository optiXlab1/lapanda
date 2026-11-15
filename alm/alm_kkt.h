#ifndef LAPANDA_ALM_KKT_H
#define LAPANDA_ALM_KKT_H

#include "alm.h"
#include <stddef.h>

/* These products reuse the existing generated oracle. No assembled matrix. */
typedef int (*alm_jv_fun)(const real_t*, const real_t*, const real_t*, const real_t*, real_t*);
typedef int (*alm_weighted_product_fun)(const real_t*, const real_t*, const real_t*, const real_t*, const real_t*, real_t*);

typedef struct {
    alm_jv_fun constraint_jv;
    alm_weighted_product_fun constraint_weighted_hvp;
    alm_weighted_product_fun constraint_weighted_vjp;
    alm_constraint_jtprod_fun constraint_theta_jtprod;
} alm_kkt_products;

typedef struct {
    unsigned int dimension;
    unsigned int free_variables;
    unsigned int active_constraints;
    unsigned int iterations;
    real_t relative_residual;
    size_t peak_workspace_bytes;
} alm_kkt_info;

/* Differentiate the original active KKT at the supplied forward point. The
 * multipliers/penalties are read-only; box faces must be parameter independent.
 * The caller chooses forward-multiplier projection policy before this call. */
int alm_backward_original_kkt(
    const alm_problem* problem,
    const alm_kkt_products* products,
    const real_t* solution,
    const real_t* multipliers,
    const real_t* penalties,
    const real_t* theta,
    const real_t* variable,
    real_t final_gamma,
    real_t tolerance,
    unsigned int max_iterations,
    real_t* gradient,
    alm_kkt_info* info);

#endif
