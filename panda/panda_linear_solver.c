#include "panda_linear_solver.h"
#include "matrix_operations.h"

#include <stdlib.h>
#include <math.h>

static size_t g_last_peak_workspace_bytes = 0;
static size_t g_persistent_workspace_bytes = 0;

size_t panda_linear_solver_get_last_peak_workspace_bytes(void)
{
    return g_last_peak_workspace_bytes;
}
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

    {
        const size_t temporary_bytes = sizeof(real_t) * ((size_t)cols * cols + cols);
        const size_t candidate = g_persistent_workspace_bytes + temporary_bytes;
        if (candidate > g_last_peak_workspace_bytes) {
            g_last_peak_workspace_bytes = candidate;
        }
    }

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
    g_last_peak_workspace_bytes = 0;
    g_persistent_workspace_bytes = 0;

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

    g_persistent_workspace_bytes = sizeof(real_t) * (size_t)3u * n;
    g_last_peak_workspace_bytes = g_persistent_workspace_bytes;

    for (i = 0; i < n; ++i) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = r[i];
    }

    bnorm = vector_norm2(b, (int)n);
    if (bnorm == 0.0) {
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
        /* CG requires positive curvature along every search direction. */
        if (pAp <= MACHINE_ACCURACY) {
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
            /* A small recursive residual is not by itself an acceptance
             * certificate. Reuse Ap for the true original-system residual. */
            rnorm = compute_relative_residual(matvec, user_data, b, x, Ap, n, bnorm);
            if (rnorm < 0.0) goto fail;
            if (relres != NULL) *relres = rnorm;
            if (iter != NULL) *iter = k + 1;
            free(r);
            free(p);
            free(Ap);
            return rnorm <= tol ? SUCCESS : FAILURE;
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

/* Paige--Saunders Lanczos/Givens short recurrence. Eight O(n) vectors;
 * acceptance always checks the explicit, unscaled relative residual. */
static real_t minres_stable_norm(const real_t* x, unsigned int n)
{
    real_t scale=0.0,ssq=1.0;
    unsigned int i;
    for(i=0;i<n;++i) {
        real_t a=fabs(x[i]);
        if(a!=0.0) {
            if(scale<a) {real_t r=scale/a;ssq=1.0+ssq*r*r;scale=a;}
            else {real_t r=a/scale;ssq+=r*r;}
        }
    }
    return scale*sqrt(ssq);
}

int panda_minres_solve(
    panda_matvec_fun matvec,void* user_data,const real_t* b,real_t* x,
    unsigned int n,real_t tol,unsigned int max_iter,real_t* relres,unsigned int* iter)
{
    real_t *buffer,*r1,*r2,*v,*y,*w,*w1,*w2,*tmp;
    real_t bn,beta,oldb,cs,sn,dbar,epsln,phibar,rr;
    unsigned int i,k,its=0;
    int status=FAILURE;
    g_last_peak_workspace_bytes=0;g_persistent_workspace_bytes=0;
    if(relres!=NULL) *relres=0.0;
    if(iter!=NULL) *iter=0;
    if(n==0) return SUCCESS;
    if(matvec==NULL||b==NULL||x==NULL||!(tol>0.0)) return FAILURE;
    set_zero(x,n);bn=minres_stable_norm(b,n);
    if(bn==0.0) return SUCCESS;
    buffer=(real_t*)calloc((size_t)8*n,sizeof(real_t));
    if(buffer==NULL) return FAILURE;
    g_last_peak_workspace_bytes=(size_t)8*n*sizeof(real_t);
    r1=buffer;r2=r1+n;v=r2+n;y=v+n;w=y+n;w1=w+n;w2=w1+n;tmp=w2+n;
    for(i=0;i<n;++i) r2[i]=b[i];
    beta=bn;oldb=0.0;cs=-1.0;sn=0.0;dbar=0.0;epsln=0.0;phibar=bn;rr=1.0;
    for(k=0;k<max_iter;++k) {
        real_t alpha,oldeps,delta,gbar,gamma,phi;
        if(!(beta>0.0)) break;
        for(i=0;i<n;++i) v[i]=r2[i]/beta;
        if(matvec(v,y,user_data)==FAILURE) goto done;
        if(k>0) for(i=0;i<n;++i) y[i]-=(beta/oldb)*r1[i];
        alpha=inner_product(v,y,(int)n);
        for(i=0;i<n;++i) y[i]-=(alpha/beta)*r2[i];
        for(i=0;i<n;++i) {r1[i]=r2[i];r2[i]=y[i];}
        oldb=beta;beta=minres_stable_norm(r2,n);
        oldeps=epsln;delta=cs*dbar+sn*alpha;gbar=sn*dbar-cs*alpha;
        epsln=sn*beta;dbar=-cs*beta;gamma=hypot(gbar,beta);
        if(!(gamma>DBL_MIN)) break;
        cs=gbar/gamma;sn=beta/gamma;phi=cs*phibar;phibar*=sn;
        for(i=0;i<n;++i) {w1[i]=w2[i];w2[i]=w[i];w[i]=(v[i]-oldeps*w1[i]-delta*w2[i])/gamma;x[i]+=phi*w[i];}
        its=k+1;
        if(fabs(phibar)<=tol*bn||beta<=DBL_MIN||its==max_iter) {
            rr=compute_relative_residual(matvec,user_data,b,x,tmp,n,bn);
            if(rr<0.0) goto done;
            if(rr<=tol) {status=SUCCESS;break;}
        }
    }
    rr=compute_relative_residual(matvec,user_data,b,x,tmp,n,bn);
    if(rr>=0.0&&rr<=tol) status=SUCCESS;
done:
    if(relres!=NULL) *relres=rr;
    if(iter!=NULL) *iter=its;
    free(buffer);return status;
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
    g_last_peak_workspace_bytes = 0;
    g_persistent_workspace_bytes = 0;

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

    g_persistent_workspace_bytes = sizeof(real_t) * (
        (size_t)(restart + 1) * n
        + (size_t)(restart + 1) * restart
        + restart
        + 2u * n);
    g_last_peak_workspace_bytes = g_persistent_workspace_bytes;

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
