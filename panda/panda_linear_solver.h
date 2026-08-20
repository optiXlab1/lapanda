#ifndef PANDA_LINEAR_SOLVER_H
#define PANDA_LINEAR_SOLVER_H

#include "../globals/globals.h"

typedef int (*panda_matvec_fun)(
    const real_t* x,
    real_t* y,
    void* user_data
);

/*
 * CG backend for symmetric positive-definite reduced systems.
 */
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
);

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
);

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
);

#endif
