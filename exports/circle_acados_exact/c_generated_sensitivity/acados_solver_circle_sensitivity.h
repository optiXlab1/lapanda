/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_circle_sensitivity_H_
#define ACADOS_SOLVER_circle_sensitivity_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define CIRCLE_SENSITIVITY_NX     3
#define CIRCLE_SENSITIVITY_NZ     0
#define CIRCLE_SENSITIVITY_NU     2
#define CIRCLE_SENSITIVITY_NP     0
#define CIRCLE_SENSITIVITY_NP_GLOBAL     13
#define CIRCLE_SENSITIVITY_NBX    0
#define CIRCLE_SENSITIVITY_NBX0   3
#define CIRCLE_SENSITIVITY_NBU    2
#define CIRCLE_SENSITIVITY_NSBX   0
#define CIRCLE_SENSITIVITY_NSBU   0
#define CIRCLE_SENSITIVITY_NSH    0
#define CIRCLE_SENSITIVITY_NSH0   0
#define CIRCLE_SENSITIVITY_NSG    0
#define CIRCLE_SENSITIVITY_NSPHI  0
#define CIRCLE_SENSITIVITY_NSHN   0
#define CIRCLE_SENSITIVITY_NSGN   0
#define CIRCLE_SENSITIVITY_NSPHIN 0
#define CIRCLE_SENSITIVITY_NSPHI0 0
#define CIRCLE_SENSITIVITY_NSBXN  0
#define CIRCLE_SENSITIVITY_NS     0
#define CIRCLE_SENSITIVITY_NS0    0
#define CIRCLE_SENSITIVITY_NSN    0
#define CIRCLE_SENSITIVITY_NG     0
#define CIRCLE_SENSITIVITY_NBXN   0
#define CIRCLE_SENSITIVITY_NGN    0
#define CIRCLE_SENSITIVITY_NY0    0
#define CIRCLE_SENSITIVITY_NY     0
#define CIRCLE_SENSITIVITY_NYN    0
#define CIRCLE_SENSITIVITY_N      12
#define CIRCLE_SENSITIVITY_NH     1
#define CIRCLE_SENSITIVITY_NHN    0
#define CIRCLE_SENSITIVITY_NH0    0
#define CIRCLE_SENSITIVITY_NPHI0  0
#define CIRCLE_SENSITIVITY_NPHI   0
#define CIRCLE_SENSITIVITY_NPHIN  0
#define CIRCLE_SENSITIVITY_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct circle_sensitivity_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    external_function_casadi p_global_precompute_fun;
    // dynamics

    external_function_external_param_casadi *discr_dyn_phi_fun;
    external_function_external_param_casadi *discr_dyn_phi_fun_jac_ut_xt;

    external_function_external_param_casadi *discr_dyn_phi_jac_p_hess_xu_p;

    external_function_external_param_casadi *discr_dyn_phi_adj_p;
    external_function_external_param_casadi *discr_dyn_phi_fun_jac_ut_xt_hess;


    // cost

    external_function_external_param_casadi *ext_cost_fun;
    external_function_external_param_casadi *ext_cost_fun_jac;
    external_function_external_param_casadi *ext_cost_fun_jac_hess;

    external_function_external_param_casadi *ext_cost_hess_xu_p;

    external_function_external_param_casadi *ext_cost_grad_p;



    external_function_external_param_casadi ext_cost_0_fun;
    external_function_external_param_casadi ext_cost_0_fun_jac;
    external_function_external_param_casadi ext_cost_0_fun_jac_hess;

    external_function_external_param_casadi ext_cost_0_hess_xu_p;

    external_function_external_param_casadi ext_cost_0_grad_p;


    external_function_external_param_casadi ext_cost_e_fun;
    external_function_external_param_casadi ext_cost_e_fun_jac;
    external_function_external_param_casadi ext_cost_e_fun_jac_hess;

    external_function_external_param_casadi ext_cost_e_hess_xu_p;

    external_function_external_param_casadi ext_cost_e_grad_p;

    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;
    external_function_external_param_casadi *nl_constr_h_fun_jac_hess;

    external_function_external_param_casadi *nl_constr_h_jac_p_hess_xu_p;

    external_function_external_param_casadi *nl_constr_h_adj_p;







} circle_sensitivity_solver_capsule;

ACADOS_SYMBOL_EXPORT circle_sensitivity_solver_capsule * circle_sensitivity_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_free_capsule(circle_sensitivity_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_create(circle_sensitivity_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_reset(circle_sensitivity_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of circle_sensitivity_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_create_with_discretization(circle_sensitivity_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_update_time_steps(circle_sensitivity_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_update_qp_solver_cond_N(circle_sensitivity_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_update_params(circle_sensitivity_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_update_params_sparse(circle_sensitivity_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_set_p_global_and_precompute_dependencies(circle_sensitivity_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_solve(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_setup_qp_matrices_and_factorize(circle_sensitivity_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_free(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void circle_sensitivity_acados_print_stats(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int circle_sensitivity_acados_custom_update(circle_sensitivity_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *circle_sensitivity_acados_get_nlp_in(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *circle_sensitivity_acados_get_nlp_out(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *circle_sensitivity_acados_get_sens_out(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *circle_sensitivity_acados_get_nlp_solver(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *circle_sensitivity_acados_get_nlp_config(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *circle_sensitivity_acados_get_nlp_opts(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *circle_sensitivity_acados_get_nlp_dims(circle_sensitivity_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *circle_sensitivity_acados_get_nlp_plan(circle_sensitivity_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_circle_sensitivity_H_
