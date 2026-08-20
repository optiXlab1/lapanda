#include "panda_linear_solver.h"
#include "matrix_operations.h"

#include <stdlib.h>
#include <math.h>

static void set_zero(real_t* x, unsigned int n)
{
    unsigned int i;

    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
    }
}

static int solve_dense_system(real_t* A, real_t* b, real_t* x, unsigned int n)
{
    real_t pivot;
    real_t factor;
    real_t tmp;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int p;

    for (k = 0; k < n; ++k) {
        p = k;
        pivot = fabs(A[k * n + k]);
        for (i = k + 1; i < n; ++i) {
            tmp = fabs(A[i * n + k]);
            if (tmp > pivot) {
                pivot = tmp;
                p = i;
            }
        }

        if (pivot < MACHINE_ACCURACY) {
            return FAILURE;
        }

        if (p != k) {
            for (j = k; j < n; ++j) {
                tmp = A[k * n + j];
                A[k * n + j] = A[p * n + j];
                A[p * n + j] = tmp;
            }
            tmp = b[k];
            b[k] = b[p];
            b[p] = tmp;
        }

        for (i = k + 1; i < n; ++i) {
            factor = A[i * n + k] / A[k * n + k];
            A[i * n + k] = 0.0;
            for (j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (i = n; i > 0; --i) {
        k = i - 1;
        tmp = b[k];
        for (j = k + 1; j < n; ++j) {
            tmp -= A[k * n + j] * x[j];
        }
        x[k] = tmp / A[k * n + k];
    }

    return SUCCESS;
}

static int gmres_least_squares(
    const real_t* H,
    real_t beta,
    real_t* y,
    unsigned int rows,
    unsigned int cols,
    unsigned int ld
)
{
    real_t* A;
    real_t* rhs;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    int status;

    A = NULL;
    rhs = NULL;

    A = (real_t*)malloc(sizeof(real_t) * cols * cols);
    if (A == NULL) goto fail;
    rhs = (real_t*)malloc(sizeof(real_t) * cols);
    if (rhs == NULL) goto fail;

    for (i = 0; i < cols; ++i) {
        rhs[i] = beta * H[i];
        for (j = 0; j < cols; ++j) {
            A[i * cols + j] = 0.0;
            for (k = 0; k < rows; ++k) {
                A[i * cols + j] += H[k * ld + i] * H[k * ld + j];
            }
        }
    }

    set_zero(y, cols);
    status = solve_dense_system(A, rhs, y, cols);

    free(A);
    free(rhs);
    return status;

fail:
    free(A);
    free(rhs);
    return FAILURE;
}

static real_t compute_relative_residual(
    panda_matvec_fun matvec,
    void* user_data,
    const real_t* b,
    const real_t* x,
    real_t* work,
    unsigned int n,
    real_t bnorm
)
{
    unsigned int i;

    if (matvec(x, work, user_data) == FAILURE) {
        return -1.0;
    }

    for (i = 0; i < n; ++i) {
        work[i] = b[i] - work[i];
    }

    return vector_norm2(work, (int)n) / bnorm;
}

int panda_spd_solve(
    panda_matvec_fun matvec,
    void* user_data,
    const real_t* b,
    real_t* x,
    unsigned int n,
    real_t tol,
    unsigned int max_iter,
    real_t* relres,
    unsigned int* iter
)
{
    real_t* r;
    real_t* p;
    real_t* Ap;
    real_t bnorm;
    real_t rr;
    real_t rr_new;
    real_t alpha;
    real_t beta;
    real_t pAp;
    real_t rnorm;
    unsigned int i;
    unsigned int k;

    r = NULL;
    p = NULL;
    Ap = NULL;

    if (n == 0) {
        if (relres != NULL) *relres = 0.0;
        if (iter != NULL) *iter = 0;
        return SUCCESS;
    }

    r = (real_t*)malloc(sizeof(real_t) * n);
    if (r == NULL) goto fail;
    p = (real_t*)malloc(sizeof(real_t) * n);
    if (p == NULL) goto fail;
    Ap = (real_t*)malloc(sizeof(real_t) * n);
    if (Ap == NULL) goto fail;

    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }

    bnorm = vector_norm2(b, (int)n);
    if (bnorm < MACHINE_ACCURACY) {
        if (relres != NULL) *relres = 0.0;
        if (iter != NULL) *iter = 0;
        free(r);
        free(p);
        free(Ap);
        return SUCCESS;
    }

    rr = inner_product(r, r, (int)n);
    rnorm = sqrt(rr) / bnorm;

    if (rnorm <= tol) {
        if (relres != NULL) *relres = rnorm;
        if (iter != NULL) *iter = 0;
        free(r);
        free(p);
        free(Ap);
        return SUCCESS;
    }

    for (k = 0; k < max_iter; ++k) {
        if (matvec(p, Ap, user_data) == FAILURE) {
            goto fail;
        }

        pAp = inner_product(p, Ap, (int)n);
        if (fabs(pAp) < MACHINE_ACCURACY) {
            if (relres != NULL) *relres = rnorm;
            if (iter != NULL) *iter = k;
            free(r);
            free(p);
            free(Ap);
            return FAILURE;
        }

        alpha = rr / pAp;

        for (i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        rr_new = inner_product(r, r, (int)n);
        rnorm = sqrt(rr_new) / bnorm;

        if (rnorm <= tol) {
            if (relres != NULL) *relres = rnorm;
            if (iter != NULL) *iter = k + 1;
            free(r);
            free(p);
            free(Ap);
            return SUCCESS;
        }

        beta = rr_new / rr;
        for (i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }

    if (relres != NULL) *relres = rnorm;
    if (iter != NULL) *iter = max_iter;

    free(r);
    free(p);
    free(Ap);
    return FAILURE;

fail:
    free(r);
    free(p);
    free(Ap);
    return FAILURE;
}

int panda_minres_solve(
    panda_matvec_fun matvec,
    void* user_data,
    const real_t* b,
    real_t* x,
    unsigned int n,
    real_t tol,
    unsigned int max_iter,
    real_t* relres,
    unsigned int* iter
)
{
    real_t* V;
    real_t* Tbar;
    real_t* y;
    real_t* work;
    real_t* x_trial;
    real_t bnorm;
    real_t beta_prev;
    real_t beta_next;
    real_t alpha;
    real_t hval;
    real_t rnorm;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int rows;
    int status;

    V = NULL;
    Tbar = NULL;
    y = NULL;
    work = NULL;
    x_trial = NULL;

    if (n == 0) {
        if (relres != NULL) *relres = 0.0;
        if (iter != NULL) *iter = 0;
        return SUCCESS;
    }

    if (max_iter == 0) {
        max_iter = n;
    }

    bnorm = vector_norm2(b, (int)n);
    if (bnorm < MACHINE_ACCURACY) {
        for (i = 0; i < n; ++i) {
            x[i] = 0.0;
        }
        if (relres != NULL) *relres = 0.0;
        if (iter != NULL) *iter = 0;
        return SUCCESS;
    }

    V = (real_t*)malloc(sizeof(real_t) * (max_iter + 1) * n);
    if (V == NULL) goto fail;
    Tbar = (real_t*)malloc(sizeof(real_t) * (max_iter + 1) * max_iter);
    if (Tbar == NULL) goto fail;
    y = (real_t*)malloc(sizeof(real_t) * max_iter);
    if (y == NULL) goto fail;
    work = (real_t*)malloc(sizeof(real_t) * n);
    if (work == NULL) goto fail;
    x_trial = (real_t*)malloc(sizeof(real_t) * n);
    if (x_trial == NULL) goto fail;

    set_zero(Tbar, (max_iter + 1) * max_iter);
    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
        V[i] = b[i] / bnorm;
    }

    beta_prev = 0.0;
    rnorm = 1.0;
    status = FAILURE;

    for (k = 0; k < max_iter; ++k) {
        if (matvec(&V[k * n], work, user_data) == FAILURE) {
            goto fail;
        }

        if (k > 0) {
            for (i = 0; i < n; ++i) {
                work[i] -= beta_prev * V[(k - 1) * n + i];
            }
        }

        alpha = inner_product(&V[k * n], work, (int)n);
        for (i = 0; i < n; ++i) {
            work[i] -= alpha * V[k * n + i];
        }

        /*
         * Full reorthogonalization keeps the stored Lanczos basis reliable
         * for this validation-oriented implementation.
         */
        for (j = 0; j <= k; ++j) {
            hval = inner_product(work, &V[j * n], (int)n);
            for (i = 0; i < n; ++i) {
                work[i] -= hval * V[j * n + i];
            }
        }

        beta_next = vector_norm2(work, (int)n);

        Tbar[k * max_iter + k] = alpha;
        if (k > 0) {
            Tbar[(k - 1) * max_iter + k] = beta_prev;
            Tbar[k * max_iter + (k - 1)] = beta_prev;
        }
        Tbar[(k + 1) * max_iter + k] = beta_next;

        if (beta_next > MACHINE_ACCURACY && k + 1 < max_iter + 1) {
            for (i = 0; i < n; ++i) {
                V[(k + 1) * n + i] = work[i] / beta_next;
            }
        }

        rows = k + 2;
        if (gmres_least_squares(Tbar, bnorm, y, rows, k + 1, max_iter) == FAILURE) {
            goto fail;
        }

        for (i = 0; i < n; ++i) {
            x_trial[i] = 0.0;
        }

        for (j = 0; j <= k; ++j) {
            for (i = 0; i < n; ++i) {
                x_trial[i] += y[j] * V[j * n + i];
            }
        }

        rnorm = compute_relative_residual(
            matvec,
            user_data,
            b,
            x_trial,
            work,
            n,
            bnorm);
        if (rnorm < 0.0) goto fail;

        for (i = 0; i < n; ++i) {
            x[i] = x_trial[i];
        }

        if (rnorm <= tol) {
            status = SUCCESS;
            k++;
            break;
        }

        if (beta_next <= MACHINE_ACCURACY) {
            status = (rnorm <= tol) ? SUCCESS : FAILURE;
            k++;
            break;
        }

        beta_prev = beta_next;
    }

    if (relres != NULL) *relres = rnorm;
    if (iter != NULL) *iter = k;

    free(V);
    free(Tbar);
    free(y);
    free(work);
    free(x_trial);
    return status;

fail:
    free(V);
    free(Tbar);
    free(y);
    free(work);
    free(x_trial);
    return FAILURE;
}

int panda_gmres_solve(
    panda_matvec_fun matvec,
    void* user_data,
    const real_t* b,
    real_t* x,
    unsigned int n,
    real_t tol,
    unsigned int max_iter,
    unsigned int restart,
    real_t* relres,
    unsigned int* iter
)
{
    real_t* V;
    real_t* H;
    real_t* y;
    real_t* w;
    real_t* r;
    real_t bnorm;
    real_t beta;
    real_t hval;
    real_t rnorm;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int cycle_iter;
    unsigned int total_iter;
    unsigned int rows;
    int status;

    V = NULL;
    H = NULL;
    y = NULL;
    w = NULL;
    r = NULL;

    if (n == 0) {
        if (relres != NULL) *relres = 0.0;
        if (iter != NULL) *iter = 0;
        return SUCCESS;
    }

    if (restart == 0 || restart > max_iter) {
        restart = max_iter;
    }

    bnorm = vector_norm2(b, (int)n);
    if (bnorm < MACHINE_ACCURACY) {
        for (i = 0; i < n; ++i) {
            x[i] = 0.0;
        }
        if (relres != NULL) *relres = 0.0;
        if (iter != NULL) *iter = 0;
        return SUCCESS;
    }

    V = (real_t*)malloc(sizeof(real_t) * (restart + 1) * n);
    if (V == NULL) goto fail;
    H = (real_t*)malloc(sizeof(real_t) * (restart + 1) * restart);
    if (H == NULL) goto fail;
    y = (real_t*)malloc(sizeof(real_t) * restart);
    if (y == NULL) goto fail;
    w = (real_t*)malloc(sizeof(real_t) * n);
    if (w == NULL) goto fail;
    r = (real_t*)malloc(sizeof(real_t) * n);
    if (r == NULL) goto fail;

    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
    }

    total_iter = 0;
    rnorm = compute_relative_residual(matvec, user_data, b, x, r, n, bnorm);
    if (rnorm < 0.0) goto fail;

    while (rnorm > tol && total_iter < max_iter) {
        beta = vector_norm2(r, (int)n);
        if (beta < MACHINE_ACCURACY) {
            rnorm = 0.0;
            break;
        }

        for (i = 0; i < n; ++i) {
            V[i] = r[i] / beta;
        }

        set_zero(H, (restart + 1) * restart);
        cycle_iter = 0;

        for (j = 0; j < restart && total_iter < max_iter; ++j) {
            if (matvec(&V[j * n], w, user_data) == FAILURE) {
                goto fail;
            }

            for (i = 0; i <= j; ++i) {
                hval = inner_product(w, &V[i * n], (int)n);
                H[i * restart + j] = hval;
                for (k = 0; k < n; ++k) {
                    w[k] -= hval * V[i * n + k];
                }
            }

            hval = vector_norm2(w, (int)n);
            H[(j + 1) * restart + j] = hval;
            if (hval > MACHINE_ACCURACY && j + 1 < restart + 1) {
                for (k = 0; k < n; ++k) {
                    V[(j + 1) * n + k] = w[k] / hval;
                }
            }

            cycle_iter = j + 1;
            total_iter++;

            rows = cycle_iter + 1;
            if (gmres_least_squares(H, beta, y, rows, cycle_iter, restart) == FAILURE) {
                goto fail;
            }

            rnorm = compute_relative_residual(matvec, user_data, b, x, r, n, bnorm);
            if (rnorm < 0.0) goto fail;

            if (hval <= MACHINE_ACCURACY || rnorm <= tol) {
                break;
            }
        }

        if (gmres_least_squares(H, beta, y, cycle_iter + 1, cycle_iter, restart) == FAILURE) {
            goto fail;
        }

        for (j = 0; j < cycle_iter; ++j) {
            for (i = 0; i < n; ++i) {
                x[i] += y[j] * V[j * n + i];
            }
        }

        rnorm = compute_relative_residual(matvec, user_data, b, x, r, n, bnorm);
        if (rnorm < 0.0) goto fail;
    }

    status = (rnorm <= tol) ? SUCCESS : FAILURE;
    if (relres != NULL) *relres = rnorm;
    if (iter != NULL) *iter = total_iter;

    free(V);
    free(H);
    free(y);
    free(w);
    free(r);
    return status;

fail:
    free(V);
    free(H);
    free(y);
    free(w);
    free(r);
    return FAILURE;
}
