#ifndef LBFGS_H
#define LBFGS_H

#include "../globals/globals.h"

int lbfgs_init(unsigned int buffer_size, unsigned int dimension_problem);
int lbfgs_cleanup(void);
int lbfgs_reset(void);

const real_t* lbfgs_get_direction(const real_t* prev_res);

int lbfgs_update(const real_t* s_new, const real_t* y_new, real_t norm_residual_map_prev);

unsigned int lbfgs_get_active_buffer_size(void);

#endif