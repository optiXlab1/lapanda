#ifndef PANDA_FORWARD_H
#define PANDA_FORWARD_H

#include "../globals/globals.h"
#include "panda_types.h"

int panda_forward_init(const panda_params* params);
int panda_forward_cleanup(void);

int panda_forward_set_initial(const real_t* x0);
real_t panda_forward_step(void);
int panda_forward_reset(void);

const real_t* panda_forward_get_solution(void);
const real_t* panda_forward_get_residual(void);
real_t panda_forward_get_gamma(void);
real_t panda_forward_get_tau(void);
real_t panda_forward_get_phi(void);
real_t panda_forward_get_f_x(void);
real_t panda_forward_get_f_z(void);
real_t panda_forward_get_g_z(void);
real_t panda_forward_get_residual_norm2(void);
unsigned char panda_forward_get_upper_ok(void);

int panda_init(const panda_params* params);
int panda_cleanup(void);
int panda_set_initial(const real_t* x0);
real_t panda_step(void);
int panda_reset(void);
const real_t* panda_get_solution(void);
const real_t* panda_get_residual(void);
real_t panda_get_gamma(void);
real_t panda_get_tau(void);
real_t panda_get_phi(void);
real_t panda_get_f_x(void);
real_t panda_get_f_z(void);
real_t panda_get_g_z(void);
real_t panda_get_residual_norm2(void);
unsigned char panda_get_upper_ok(void);

#endif
