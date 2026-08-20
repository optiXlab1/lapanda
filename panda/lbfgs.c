#include "lbfgs.h"

#include "matrix_operations.h"

#include <stddef.h>
#include <stdlib.h>

static real_t** S = NULL;
static real_t** Y = NULL;

static real_t* S_data = NULL;
static real_t* Y_data = NULL;

static real_t* alpha_arr = NULL;
static real_t* rho_arr = NULL;

static real_t* direction = NULL;

static unsigned int active_buffer_size = 0;
static unsigned int buffer_size = 0;
static unsigned int dimension = 0;

static real_t hessian_estimate = 1.0;

static void shift_pairs(void);
static int reset_direction(void);
static int valid_update_pair(const real_t* s_new, const real_t* y_new, real_t norm_residual_map_prev);

int lbfgs_init(unsigned int buffer_size_, unsigned int dimension_problem)
{
    unsigned int i;

    buffer_size = buffer_size_;
    dimension = dimension_problem;
    active_buffer_size = 0;
    hessian_estimate = 1.0;

    S_data = malloc(sizeof(real_t) * dimension * (buffer_size + 1));
    if (S_data == NULL) goto fail_1;

    S = malloc(sizeof(real_t*) * (buffer_size + 1));
    if (S == NULL) goto fail_2;

    Y_data = malloc(sizeof(real_t) * dimension * (buffer_size + 1));
    if (Y_data == NULL) goto fail_3;

    Y = malloc(sizeof(real_t*) * (buffer_size + 1));
    if (Y == NULL) goto fail_4;

    alpha_arr = malloc(sizeof(real_t) * buffer_size);
    if (alpha_arr == NULL) goto fail_5;

    rho_arr = malloc(sizeof(real_t) * buffer_size);
    if (rho_arr == NULL) goto fail_6;

    direction = malloc(sizeof(real_t) * dimension);
    if (direction == NULL) goto fail_7;

    for (i = 0; i < buffer_size + 1; ++i) {
        S[i] = S_data + i * dimension;
        Y[i] = Y_data + i * dimension;
    }

    return SUCCESS;

fail_7:
    free(rho_arr);
fail_6:
    free(alpha_arr);
fail_5:
    free(Y);
fail_4:
    free(Y_data);
fail_3:
    free(S);
fail_2:
    free(S_data);
fail_1:
    return FAILURE;
}

int lbfgs_cleanup(void)
{
    free(S_data);
    free(S);
    free(Y_data);
    free(Y);
    free(alpha_arr);
    free(rho_arr);
    free(direction);

    S_data = NULL;
    S = NULL;
    Y_data = NULL;
    Y = NULL;
    alpha_arr = NULL;
    rho_arr = NULL;
    direction = NULL;

    active_buffer_size = 0;
    buffer_size = 0;
    dimension = 0;
    hessian_estimate = 1.0;

    return SUCCESS;
}

int lbfgs_reset(void)
{
    active_buffer_size = 0;
    hessian_estimate = 1.0;
    return SUCCESS;
}

unsigned int lbfgs_get_active_buffer_size(void)
{
    return active_buffer_size;
}

const real_t* lbfgs_get_direction(const real_t* prev_res)
{
    unsigned int m;
    int i;
    real_t beta;

    if (vector_norm2(prev_res, dimension) < MACHINE_ACCURACY) {
        reset_direction();
        return direction;
    }

    if (active_buffer_size == 0) {
        vector_copy(prev_res, direction, dimension);
        vector_real_mul(direction, dimension, -1.0);
        return direction;
    }

    m = (active_buffer_size < buffer_size) ? active_buffer_size : buffer_size;

    vector_copy(prev_res, direction, dimension);

    for (i = 0; i < (int)m; ++i) {
        rho_arr[i] = 1.0 / inner_product(Y[i], S[i], dimension);
        alpha_arr[i] = rho_arr[i] * inner_product(S[i], direction, dimension);
        vector_add_ntimes(direction, Y[i], dimension, -alpha_arr[i]);
    }

    vector_real_mul(direction, dimension, hessian_estimate);

    for (i = (int)m - 1; i >= 0; --i) {
        beta = rho_arr[i] * inner_product(Y[i], direction, dimension);
        vector_add_ntimes(direction, S[i], dimension, alpha_arr[i] - beta);
    }

    vector_real_mul(direction, dimension, -1.0);
    return direction;
}

int lbfgs_update(const real_t* s_new, const real_t* y_new, real_t norm_residual_map_prev)
{
    real_t sy;
    real_t yy;

    if (valid_update_pair(s_new, y_new, norm_residual_map_prev) == FAILURE) {
        return FAILURE;
    }

    vector_copy(s_new, S[buffer_size], dimension);
    vector_copy(y_new, Y[buffer_size], dimension);

    shift_pairs();

    sy = inner_product(S[0], Y[0], dimension);
    yy = inner_product(Y[0], Y[0], dimension);

    hessian_estimate = sy / yy;

    active_buffer_size++;
    return SUCCESS;
}

static int valid_update_pair(const real_t* s_new, const real_t* y_new, real_t norm_residual_map_prev)
{
    real_t sy;
    real_t ss;

    sy = inner_product(s_new, y_new, dimension);
    ss = inner_product(s_new, s_new, dimension);

    if (ss < MACHINE_ACCURACY) {
        return FAILURE;
    }

    if (sy / ss < (1e-12) * norm_residual_map_prev) {
        return FAILURE;
    }

    return SUCCESS;
}

static void shift_pairs(void)
{
    real_t* tmp_s;
    real_t* tmp_y;
    unsigned int i;

    tmp_s = S[buffer_size];
    tmp_y = Y[buffer_size];

    for (i = buffer_size; i > 0; --i) {
        S[i] = S[i - 1];
        Y[i] = Y[i - 1];
    }

    S[0] = tmp_s;
    Y[0] = tmp_y;
}

static int reset_direction(void)
{
    unsigned int i;
    for (i = 0; i < dimension; ++i) {
        direction[i] = 0.0;
    }
    return SUCCESS;
}