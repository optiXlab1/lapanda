#include "../globals/globals.h"
#include "../panda/panda_linear_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const real_t* A;
    unsigned int n;
} dense_matvec_data;

static int dense_matvec(const real_t* x, real_t* y, void* user_data)
{
    dense_matvec_data* data;
    unsigned int i;
    unsigned int j;

    data = (dense_matvec_data*)user_data;

    for (i = 0; i < data->n; ++i) {
        y[i] = 0.0;
        for (j = 0; j < data->n; ++j) {
            y[i] += data->A[i * data->n + j] * x[j];
        }
    }

    return SUCCESS;
}

static real_t max_abs_error(const real_t* x, const real_t* ref, unsigned int n)
{
    real_t err;
    real_t err_i;
    unsigned int i;

    err = 0.0;
    for (i = 0; i < n; ++i) {
        err_i = fabs(x[i] - ref[i]);
        if (err_i > err) {
            err = err_i;
        }
    }

    return err;
}

static real_t relative_residual(
    const real_t* A,
    const real_t* b,
    const real_t* x,
    unsigned int n
)
{
    real_t num;
    real_t den;
    real_t ri;
    unsigned int i;
    unsigned int j;

    num = 0.0;
    den = 0.0;
    for (i = 0; i < n; ++i) {
        ri = -b[i];
        for (j = 0; j < n; ++j) {
            ri += A[i * n + j] * x[j];
        }
        num += ri * ri;
        den += b[i] * b[i];
    }

    if (den <= MACHINE_ACCURACY) {
        den = 1.0;
    }

    return sqrt(num / den);
}

static int dense_reference_solve(
    const real_t* A_in,
    const real_t* b_in,
    real_t* x,
    unsigned int n
)
{
    real_t* A;
    real_t* b;
    real_t pivot;
    real_t best;
    real_t tmp;
    real_t factor;
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int p;

    A = NULL;
    b = NULL;

    A = (real_t*)malloc(sizeof(real_t) * n * n);
    if (A == NULL) goto fail;
    b = (real_t*)malloc(sizeof(real_t) * n);
    if (b == NULL) goto fail;

    for (i = 0; i < n * n; ++i) {
        A[i] = A_in[i];
    }
    for (i = 0; i < n; ++i) {
        b[i] = b_in[i];
        x[i] = 0.0;
    }

    for (k = 0; k < n; ++k) {
        p = k;
        best = fabs(A[k * n + k]);
        for (i = k + 1; i < n; ++i) {
            pivot = fabs(A[i * n + k]);
            if (pivot > best) {
                best = pivot;
                p = i;
            }
        }

        if (best < 1e-14) goto fail;

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

    free(A);
    free(b);
    return SUCCESS;

fail:
    free(A);
    free(b);
    return FAILURE;
}

static void build_spd_system(real_t* A, real_t* b, unsigned int n)
{
    unsigned int i;
    unsigned int j;

    for (i = 0; i < n; ++i) {
        b[i] = 1.0 + 0.25 * (real_t)i;
        for (j = 0; j < n; ++j) {
            if (i == j) {
                A[i * n + j] = 6.0 + 0.5 * (real_t)i;
            } else {
                A[i * n + j] = 1.0 / (1.0 + fabs((real_t)i - (real_t)j));
            }
        }
    }
}

static void build_symmetric_indefinite_system(real_t* A, real_t* b, unsigned int n)
{
    static const real_t diag[10] = {
        -4.0, -2.0, -0.75, 0.5, 1.25, 2.0, 3.0, 4.0, 5.0, 6.0
    };
    unsigned int i;
    unsigned int j;

    for (i = 0; i < n; ++i) {
        b[i] = (i % 2 == 0) ? 1.0 + 0.1 * (real_t)i : -0.5 - 0.2 * (real_t)i;
        for (j = 0; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    for (i = 0; i < n; ++i) {
        A[i * n + i] = diag[i];
    }
    for (i = 0; i + 1 < n; ++i) {
        A[i * n + i + 1] = 0.25;
        A[(i + 1) * n + i] = 0.25;
    }
    for (i = 0; i + 2 < n; ++i) {
        A[i * n + i + 2] = -0.1;
        A[(i + 2) * n + i] = -0.1;
    }
}

static void build_nonsymmetric_system(real_t* A, real_t* b, unsigned int n)
{
    unsigned int i;
    unsigned int j;

    for (i = 0; i < n; ++i) {
        b[i] = 0.5 + 0.3 * (real_t)i;
        for (j = 0; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    for (i = 0; i < n; ++i) {
        A[i * n + i] = 3.0 + 0.2 * (real_t)i;
        if (i + 1 < n) {
            A[i * n + i + 1] = -0.7;
        }
        if (i > 0) {
            A[i * n + i - 1] = 0.3;
        }
        if (i + 3 < n) {
            A[i * n + i + 3] = 0.15;
        }
    }
}

static int run_solver_case(
    const char* name,
    int solver,
    const real_t* A,
    const real_t* b,
    unsigned int n,
    real_t tol
)
{
    dense_matvec_data data;
    real_t* x;
    real_t* ref;
    real_t relres;
    real_t actual_relres;
    real_t err;
    unsigned int iter;
    int status;

    x = NULL;
    ref = NULL;
    data.A = A;
    data.n = n;

    x = (real_t*)malloc(sizeof(real_t) * n);
    if (x == NULL) goto fail;
    ref = (real_t*)malloc(sizeof(real_t) * n);
    if (ref == NULL) goto fail;

    if (dense_reference_solve(A, b, ref, n) == FAILURE) {
        goto fail;
    }

    if (solver == 0) {
        status = panda_minres_solve(
            dense_matvec,
            &data,
            b,
            x,
            n,
            tol,
            2 * n,
            &relres,
            &iter);
    } else if (solver == 1) {
        status = panda_gmres_solve(
            dense_matvec,
            &data,
            b,
            x,
            n,
            tol,
            3 * n,
            4,
            &relres,
            &iter);
    } else {
        status = panda_spd_solve(
            dense_matvec,
            &data,
            b,
            x,
            n,
            tol,
            2 * n,
            &relres,
            &iter);
    }

    err = max_abs_error(x, ref, n);
    actual_relres = relative_residual(A, b, x, n);

    if (status == FAILURE || err > 1e-7 || actual_relres > 1e-9) {
        printf("%s failed: status=%d relres=%.3e actual=%.3e iter=%u err=%.3e\n",
               name,
               status,
               (double)relres,
               (double)actual_relres,
               iter,
               (double)err);
        free(x);
        free(ref);
        return FAILURE;
    }

    printf("%s passed: relres=%.3e iter=%u err=%.3e\n",
           name, (double)actual_relres, iter, (double)err);

    free(x);
    free(ref);
    return SUCCESS;

fail:
    free(x);
    free(ref);
    return FAILURE;
}

int main(void)
{
    const unsigned int n = 10;
    real_t A[100];
    real_t b[10];
    real_t x[10];
    real_t relres;
    unsigned int iter;
    dense_matvec_data data;

    build_spd_system(A, b, n);
    if (run_solver_case("CG SPD n=10", 2, A, b, n, 1e-12) == FAILURE) {
        return 1;
    }
    if (run_solver_case("MINRES SPD n=10", 0, A, b, n, 1e-12) == FAILURE) {
        return 1;
    }

    build_symmetric_indefinite_system(A, b, n);
    data.A = A;
    data.n = n;
    if (panda_spd_solve(
            dense_matvec, &data, b, x, n, 1e-12, 2 * n, &relres, &iter)
        != FAILURE) {
        printf("CG should reject symmetric-indefinite curvature\n");
        return 1;
    }
    if (run_solver_case("MINRES symmetric-indefinite n=10", 0, A, b, n, 1e-12) == FAILURE) {
        return 1;
    }

    build_nonsymmetric_system(A, b, n);
    if (run_solver_case("GMRES nonsymmetric n=10 restart=4", 1, A, b, n, 1e-12) == FAILURE) {
        return 1;
    }

    return 0;
}
