#include "panda_backward.h"
#include "panda_linear_solver.h"
#include "matrix_operations.h"
#include "function_evaluator.h"

#include <stdlib.h>
#include <math.h>

static size_t g_last_peak_workspace_bytes = 0;
static panda_backward_solver_type g_last_solver_used = PANDA_BACKWARD_SOLVER_AUTO;
static unsigned char g_last_fallback_used = FALSE;

size_t panda_backward_get_last_peak_workspace_bytes(void)
{
    return g_last_peak_workspace_bytes;
}

panda_backward_solver_type panda_backward_get_last_solver_used(void)
{
    return g_last_solver_used;
}

unsigned char panda_backward_get_last_fallback_used(void)
{
    return g_last_fallback_used;
}
typedef struct {
    const real_t* u_star;
    const unsigned int* Fidx;
    unsigned int nf;
    unsigned int n;
    real_t* x_full;
    real_t* Hx_full;
} hff_operator_data;

static int hff_matvec(const real_t* xF, real_t* yF, void* user_data)
{
    hff_operator_data* data;
    unsigned int i;

    data = (hff_operator_data*)user_data;

    for (i = 0; i < data->n; ++i) {
        data->x_full[i] = 0.0;
    }

    for (i = 0; i < data->nf; ++i) {
        data->x_full[data->Fidx[i]] = xF[i];
    }

    if (function_evaluator_hvp(
            data->x_full,
            data->u_star,
            data->Hx_full) == FAILURE) {
        return FAILURE;
    }

    for (i = 0; i < data->nf; ++i) {
        yF[i] = data->Hx_full[data->Fidx[i]];
    }

    return SUCCESS;
}

static real_t pseudo_rand_unit(unsigned int* seed)
{
    *seed = (*seed * 1103515245u) + 12345u;
    return ((real_t)((*seed / 65536u) % 32768u) / 16384.0) - 1.0;
}

static int check_symmetry(
    panda_matvec_fun matvec,
    void* user_data,
    unsigned int n,
    unsigned int num_tests,
    real_t tol,
    unsigned char* is_symmetric,
    real_t* sym_error
)
{
    real_t* a;
    real_t* b;
    real_t* Aa;
    real_t* Ab;
    real_t lhs;
    real_t rhs;
    real_t denom;
    real_t err_i;
    unsigned int i;
    unsigned int k;
    unsigned int seed;

    a = NULL;
    b = NULL;
    Aa = NULL;
    Ab = NULL;

    if (n == 0) {
        *is_symmetric = TRUE;
        *sym_error = 0.0;
        return SUCCESS;
    }

    a = (real_t*)malloc(sizeof(real_t) * n);
    if (a == NULL) goto fail;
    b = (real_t*)malloc(sizeof(real_t) * n);
    if (b == NULL) goto fail;
    Aa = (real_t*)malloc(sizeof(real_t) * n);
    if (Aa == NULL) goto fail;
    Ab = (real_t*)malloc(sizeof(real_t) * n);
    if (Ab == NULL) goto fail;

    seed = 17u;
    *sym_error = 0.0;

    for (k = 0; k < num_tests; ++k) {
        for (i = 0; i < n; ++i) {
            a[i] = pseudo_rand_unit(&seed);
            b[i] = pseudo_rand_unit(&seed);
        }

        if (matvec(a, Aa, user_data) == FAILURE) goto fail;
        if (matvec(b, Ab, user_data) == FAILURE) goto fail;

        lhs = inner_product(a, Ab, (int)n);
        rhs = inner_product(b, Aa, (int)n);
        denom = fabs(lhs);
        if (fabs(rhs) > denom) denom = fabs(rhs);
        if (denom < 1.0) denom = 1.0;

        err_i = fabs(lhs - rhs) / denom;
        if (err_i > *sym_error) {
            *sym_error = err_i;
        }
    }

    *is_symmetric = (*sym_error <= tol) ? TRUE : FALSE;

    free(a);
    free(b);
    free(Aa);
    free(Ab);
    return SUCCESS;

fail:
    free(a);
    free(b);
    free(Aa);
    free(Ab);
    return FAILURE;
}

int panda_backward_compute(
    const panda_backward_options* opts,
    const real_t* u_star,
    real_t gamma,
    real_t* dLdtheta,
    real_t* final_residual,
    unsigned int* iterations
)
{
    real_t L;
    real_t tol;
    real_t sym_tol;
    unsigned int max_iter;
    unsigned int restart;
    unsigned int sym_num_tests;
    unsigned int n;
    unsigned int ntheta;
    unsigned int i;
    unsigned int nf;
    int status;
    real_t relres;
    real_t sym_error;
    unsigned int lin_iter;
    unsigned char is_symmetric;
    unsigned char recover_active;
    panda_backward_solver_type force_solver;
    panda_backward_solver_type solver_used;
    size_t outer_workspace_bytes;

    real_t* B;
    real_t* J;
    real_t* x;
    real_t* v;
    real_t* rhsF;
    real_t* xF;
    unsigned int* Fidx;
    hff_operator_data op_data;

    if (u_star == NULL || dLdtheta == NULL) {
        return FAILURE;
    }

    n = function_evaluator_get_n();
    ntheta = function_evaluator_get_ntheta();
    if (n == 0 || ntheta == 0) {
        return FAILURE;
    }

    tol = 1e-10;
    max_iter = 200;
    restart = 40;
    sym_tol = 1e-8;
    sym_num_tests = 5;
    force_solver = PANDA_BACKWARD_SOLVER_AUTO;
    recover_active = TRUE;
    if (opts != NULL) {
        if (opts->tol > 0.0) tol = opts->tol;
        if (opts->max_iter > 0) max_iter = opts->max_iter;
        if (opts->restart > 0) restart = opts->restart;
        if (opts->sym_tol > 0.0) sym_tol = opts->sym_tol;
        if (opts->sym_num_tests > 0) sym_num_tests = opts->sym_num_tests;
        force_solver = opts->force_solver;
        recover_active = opts->recover_active;
    }

    B = NULL;
    J = NULL;
    x = NULL;
    v = NULL;
    rhsF = NULL;
    xF = NULL;
    Fidx = NULL;
    op_data.x_full = NULL;
    op_data.Hx_full = NULL;
    g_last_peak_workspace_bytes = 0;
    g_last_solver_used = PANDA_BACKWARD_SOLVER_AUTO;
    g_last_fallback_used = FALSE;
    outer_workspace_bytes = 0;

    B = (real_t*)malloc(sizeof(real_t) * n);
    if (B == NULL) goto fail;
    J = (real_t*)malloc(sizeof(real_t) * n);
    if (J == NULL) goto fail;
    x = (real_t*)malloc(sizeof(real_t) * n);
    if (x == NULL) goto fail;
    v = (real_t*)malloc(sizeof(real_t) * n);
    if (v == NULL) goto fail;
    Fidx = (unsigned int*)malloc(sizeof(unsigned int) * n);
    if (Fidx == NULL) goto fail;

    if (function_evaluator_loss_grad(
            u_star,
            &L,
            B) == FAILURE) {
        goto fail;
    }

    if (function_evaluator_jprox(
            u_star,
            gamma,
            J) == FAILURE) {
        goto fail;
    }

    nf = 0;
    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
        if (fabs(J[i]) > 1e-12) {
            Fidx[nf] = i;
            nf++;
        }
    }

    if (nf > 0) {
        rhsF = (real_t*)malloc(sizeof(real_t) * nf);
        if (rhsF == NULL) goto fail;
        xF = (real_t*)malloc(sizeof(real_t) * nf);
        if (xF == NULL) goto fail;
        op_data.x_full = (real_t*)malloc(sizeof(real_t) * n);
        if (op_data.x_full == NULL) goto fail;
        op_data.Hx_full = (real_t*)malloc(sizeof(real_t) * n);
        if (op_data.Hx_full == NULL) goto fail;

        for (i = 0; i < nf; ++i) {
            rhsF[i] = B[Fidx[i]];
            xF[i] = 0.0;
        }

        op_data.u_star = u_star;
        op_data.Fidx = Fidx;
        op_data.nf = nf;
        op_data.n = n;

        if (force_solver == PANDA_BACKWARD_SOLVER_CG) {
            is_symmetric = TRUE;
            sym_error = 0.0;
        } else if (force_solver == PANDA_BACKWARD_SOLVER_MINRES) {
            is_symmetric = TRUE;
            sym_error = 0.0;
        } else if (force_solver == PANDA_BACKWARD_SOLVER_GMRES) {
            is_symmetric = FALSE;
            sym_error = 0.0;
        } else {
            if (check_symmetry(
                    hff_matvec,
                    &op_data,
                    nf,
                    sym_num_tests,
                    sym_tol,
                    &is_symmetric,
                    &sym_error) == FAILURE) {
                goto fail;
            }
        }

        if (force_solver == PANDA_BACKWARD_SOLVER_CG ||
            (force_solver == PANDA_BACKWARD_SOLVER_AUTO && is_symmetric == TRUE)) {
            /* The local ALM sensitivity operator is expected to be SPD under
             * SOSC for a sufficiently large penalty.  Fall back when the
             * observed Krylov directions do not have positive curvature. */
            solver_used = PANDA_BACKWARD_SOLVER_CG;
            status = panda_spd_solve(
                hff_matvec,
                &op_data,
                rhsF,
                xF,
                nf,
                tol,
                max_iter,
                &relres,
                &lin_iter);
            if (status == FAILURE) {
                g_last_fallback_used = TRUE;
                solver_used = PANDA_BACKWARD_SOLVER_MINRES;
                status = panda_minres_solve(
                    hff_matvec,
                    &op_data,
                    rhsF,
                    xF,
                    nf,
                    tol,
                    max_iter,
                    &relres,
                    &lin_iter);
            }
        } else if (is_symmetric == TRUE) {
            solver_used = PANDA_BACKWARD_SOLVER_MINRES;
            status = panda_minres_solve(
                hff_matvec,
                &op_data,
                rhsF,
                xF,
                nf,
                tol,
                max_iter,
                &relres,
                &lin_iter);
        } else {
            solver_used = PANDA_BACKWARD_SOLVER_GMRES;
            status = panda_gmres_solve(
                hff_matvec,
                &op_data,
                rhsF,
                xF,
                nf,
                tol,
                max_iter,
                restart,
                &relres,
                &lin_iter);
        }

        outer_workspace_bytes = sizeof(real_t) * ((size_t)6u * n + (size_t)2u * nf)
            + sizeof(unsigned int) * (size_t)n;
        g_last_peak_workspace_bytes = outer_workspace_bytes
            + panda_linear_solver_get_last_peak_workspace_bytes();
        g_last_solver_used = solver_used;

        for (i = 0; i < nf; ++i) {
            x[Fidx[i]] = xF[i];
        }

        if (recover_active == TRUE && nf < n) {
            if (function_evaluator_hvp(
                    x,
                    u_star,
                    op_data.Hx_full) == FAILURE) {
                goto fail;
            }

            for (i = 0; i < n; ++i) {
                if (fabs(J[i]) <= 1e-12) {
                    x[i] = gamma * (B[i] - op_data.Hx_full[i]);
                }
            }
        }
    } else {
        status = SUCCESS;
        relres = 0.0;
        lin_iter = 0;
        solver_used = PANDA_BACKWARD_SOLVER_AUTO;
        is_symmetric = TRUE;
        sym_error = 0.0;
        g_last_peak_workspace_bytes = sizeof(real_t) * (size_t)4u * n
            + sizeof(unsigned int) * (size_t)n;
        g_last_solver_used = solver_used;
    }

    if (status == FAILURE) {
        goto fail;
    }

    for (i = 0; i < n; ++i) {
        v[i] = -J[i] * x[i];
    }

    for (i = 0; i < ntheta; ++i) {
        dLdtheta[i] = 0.0;
    }

    if (function_evaluator_vjp(
            v,
            u_star,
            dLdtheta) == FAILURE) {
        goto fail;
    }
    
    if (final_residual != NULL) {
        *final_residual = relres;
    }
    if (iterations != NULL) {
        *iterations = lin_iter;
    }

    (void)is_symmetric;
    (void)sym_error;
    (void)gamma;

    free(B);
    free(J);
    free(x);
    free(v);
    free(rhsF);
    free(xF);
    free(Fidx);
    free(op_data.x_full);
    free(op_data.Hx_full);

    return SUCCESS;

fail:
    free(B);
    free(J);
    free(x);
    free(v);
    free(rhsF);
    free(xF);
    free(Fidx);
    free(op_data.x_full);
    free(op_data.Hx_full);
    return FAILURE;
}
