#include "mex.h"

#include "alm.h"
#include "optimizer.h"

#include <math.h>
#include <string.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef long long casadi_int;
typedef int (*casadi_eval_fun)(const double**, double**, casadi_int*, double*, int);
typedef int (*casadi_work_fun)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);

typedef struct {
    casadi_eval_fun eval;
    casadi_int* iw;
    double* w;
} casadi_function;

typedef struct {
    void* library;
    casadi_function cost_grad;
    casadi_function hvp;
    casadi_function vjp;
    casadi_function loss_grad;
    casadi_function constraint;
    casadi_function constraint_jtprod;
    casadi_function constraint_jv;
    casadi_function constraint_weighted_hvp;
    casadi_function constraint_weighted_vjp;
    casadi_function constraint_theta_jtprod;
    unsigned int n;
    unsigned int ntheta;
    unsigned int nvar;
    unsigned int ncon;
    double* box_lower;
    double* box_upper;
    int has_lower;
    int has_upper;
} dynamic_context;

static dynamic_context* g_ctx = 0;
static double g_constraint_penalty_scale = 1.0;
static double g_constraint_penalty_max = 0.0;

static void* open_library(const char* path)
{
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void close_library(void* library)
{
    if (library == 0) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)library);
#else
    dlclose(library);
#endif
}

static void* load_symbol(void* library, const char* name, int required)
{
    void* ptr;
#ifdef _WIN32
    ptr = (void*)GetProcAddress((HMODULE)library, name);
#else
    ptr = dlsym(library, name);
#endif
    if (ptr == 0 && required) {
        mexErrMsgIdAndTxt("lapanda:oracle", "Generated oracle is missing symbol: %s.", name);
    }
    return ptr;
}

static void load_function(void* library, casadi_function* fun, const char* name, int required)
{
    char work_name[256];
    casadi_work_fun work_fun;
    casadi_int sz_arg;
    casadi_int sz_res;
    casadi_int sz_iw;
    casadi_int sz_w;

    fun->eval = (casadi_eval_fun)load_symbol(library, name, required);
    fun->iw = 0;
    fun->w = 0;
    if (fun->eval == 0) return;

    if (strlen(name) + 6 >= sizeof(work_name)) {
        mexErrMsgIdAndTxt("lapanda:oracle", "CasADi function name is too long.");
    }
    strcpy(work_name, name);
    strcat(work_name, "_work");
    work_fun = (casadi_work_fun)load_symbol(library, work_name, 0);
    if (work_fun == 0) return;

    sz_arg = 0;
    sz_res = 0;
    sz_iw = 0;
    sz_w = 0;
    if (work_fun(&sz_arg, &sz_res, &sz_iw, &sz_w) != 0) {
        mexErrMsgIdAndTxt("lapanda:oracle", "Failed to query CasADi work sizes for %s.", name);
    }
    if (sz_iw > 0) fun->iw = (casadi_int*)mxCalloc((mwSize)sz_iw, sizeof(casadi_int));
    if (sz_w > 0) fun->w = (double*)mxCalloc((mwSize)sz_w, sizeof(double));
}

static void free_function(casadi_function* fun)
{
    if (fun->iw != 0) mxFree(fun->iw);
    if (fun->w != 0) mxFree(fun->w);
    fun->iw = 0;
    fun->w = 0;
    fun->eval = 0;
}

static int call_casadi(casadi_function* fun, const double** arg, double** res)
{
    if (fun == 0 || fun->eval == 0) return FAILURE;
    return fun->eval(arg, res, fun->iw, fun->w, 0);
}

static double get_option(const mxArray* options, const char* name, double fallback)
{
    const mxArray* field;
    if (options == 0 || !mxIsStruct(options)) return fallback;
    field = mxGetField(options, 0, name);
    if (field == 0 || mxIsEmpty(field)) return fallback;
    return mxGetScalar(field);
}

static double get_required_scalar(const mxArray* s, const char* name)
{
    const mxArray* field = mxGetField(s, 0, name);
    if (field == 0 || mxIsEmpty(field)) {
        mexErrMsgIdAndTxt("lapanda:meta", "meta.%s is required.", name);
    }
    return mxGetScalar(field);
}

static char* get_required_string(const mxArray* s, const char* name)
{
    char* out;
    const mxArray* field = mxGetField(s, 0, name);
    if (field == 0 || !mxIsChar(field)) {
        mexErrMsgIdAndTxt("lapanda:meta", "meta.%s must be a string.", name);
    }
    out = mxArrayToString(field);
    if (out == 0) {
        mexErrMsgIdAndTxt("lapanda:meta", "Failed to read meta.%s.", name);
    }
    return out;
}

static void check_vector(const mxArray* value, mwSize expected, const char* name)
{
    if (!mxIsDouble(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != expected) {
        mexErrMsgIdAndTxt("lapanda:badInput", "%s has an unexpected size.", name);
    }
}

static double* copy_optional_vector(const mxArray* meta, const char* name, unsigned int expected, int* present)
{
    const mxArray* field = mxGetField(meta, 0, name);
    double* out;
    mwSize i;
    if (field == 0 || mxIsEmpty(field)) {
        *present = 0;
        return 0;
    }
    check_vector(field, expected, name);
    out = (double*)mxCalloc(expected > 0 ? expected : 1, sizeof(double));
    for (i = 0; i < expected; ++i) {
        out[i] = mxGetPr(field)[i];
    }
    *present = 1;
    return out;
}

static double* copy_option_vector(const mxArray* options, const char* name, unsigned int expected)
{
    const mxArray* field;
    double* out;
    mwSize i;

    if (options == 0 || !mxIsStruct(options)) return 0;
    field = mxGetField(options, 0, name);
    if (field == 0 || mxIsEmpty(field)) return 0;
    check_vector(field, expected, name);
    out = (double*)mxCalloc(expected > 0 ? expected : 1, sizeof(double));
    for (i = 0; i < expected; ++i) {
        out[i] = mxGetPr(field)[i];
    }
    return out;
}

static void fill_solver_params(struct solver_parameters* params, const mxArray* options)
{
    params->max_iterations = (unsigned int)get_option(options, "solver_max_iterations", 600.0);
    params->tolerance = get_option(options, "solver_tolerance", 1e-4);
    params->buffer_size = (unsigned int)get_option(options, "solver_buffer_size", 10.0);
    params->max_stable_iter = (unsigned int)get_option(options, "solver_max_stable_iter", 0.0);
    params->verbose = get_option(options, "verbose", 0.0) != 0.0;
}

static void fill_backward_params(struct backward_parameters* params, const mxArray* options)
{
    params->enable = get_option(options, "backward_enable", 0.0) != 0.0;
    params->tolerance = get_option(options, "backward_tolerance", 1e-4);
    params->max_iterations = (unsigned int)get_option(options, "backward_max_iterations", 50.0);
    g_constraint_penalty_scale = get_option(options, "constraint_penalty_scale", 1.0);
    g_constraint_penalty_max = get_option(options, "constraint_penalty_max", 0.0);
}

static void fill_alm_params(alm_parameters* params, const mxArray* options)
{
    params->max_iterations = (unsigned int)get_option(options, "alm_max_iterations", 20.0);
    params->tolerance = get_option(options, "alm_tolerance", 1e-4);
    params->initial_penalty = get_option(options, "alm_initial_penalty", 10.0);
    params->penalty_update_factor = get_option(options, "alm_penalty_update_factor", 10.0);
    params->max_penalty = get_option(options, "alm_max_penalty", 0.0);
    params->sufficient_decrease_factor = get_option(options, "alm_sufficient_decrease_factor", 0.25);
    params->verbose = get_option(options, "verbose", 0.0) != 0.0;
    params->warm_start_inner = get_option(options, "alm_warm_start_inner", 1.0) != 0.0;
}

static double dynamic_cost_gradient(
    const double* input,
    const double* theta,
    const double* variable,
    double* output_gradient
)
{
    double value = 0.0;
    const double* arg[3];
    double* res[2];
    arg[0] = input;
    arg[1] = theta;
    arg[2] = variable;
    res[0] = &value;
    res[1] = output_gradient;
    (void)call_casadi(&g_ctx->cost_grad, arg, res);
    return value;
}

static double dynamic_prox(double* input, double gamma, const double* theta, const double* variable)
{
    unsigned int i;
    (void)gamma;
    (void)theta;
    (void)variable;
    for (i = 0; i < g_ctx->n; ++i) {
        if (g_ctx->has_lower && input[i] < g_ctx->box_lower[i]) input[i] = g_ctx->box_lower[i];
        if (g_ctx->has_upper && input[i] > g_ctx->box_upper[i]) input[i] = g_ctx->box_upper[i];
    }
    return 0.0;
}

static int dynamic_jprox(
    const double* u_star,
    double gamma,
    const double* theta,
    const double* variable,
    double* J
)
{
    unsigned int i;
    const double active_tol = 1e-10;
    (void)gamma;
    (void)theta;
    (void)variable;
    for (i = 0; i < g_ctx->n; ++i) {
        J[i] = 1.0;
        if (g_ctx->has_lower && u_star[i] <= g_ctx->box_lower[i] + active_tol) J[i] = 0.0;
        if (g_ctx->has_upper && u_star[i] >= g_ctx->box_upper[i] - active_tol) J[i] = 0.0;
    }
    return SUCCESS;
}

static int dynamic_hvp(
    const double* v,
    const double* u_star,
    const double* theta,
    const double* variable,
    double* Hv
)
{
    const double* arg[4];
    double* res[1];
    arg[0] = v;
    arg[1] = u_star;
    arg[2] = theta;
    arg[3] = variable;
    res[0] = Hv;
    return call_casadi(&g_ctx->hvp, arg, res) == 0 ? SUCCESS : FAILURE;
}

static int dynamic_loss_grad(
    const double* u_star,
    const double* theta,
    const double* variable,
    double* L,
    double* grad_u_L
)
{
    const double* arg[3];
    double* res[2];
    if (g_ctx->loss_grad.eval == 0) {
        *L = 0.0;
        memset(grad_u_L, 0, sizeof(double) * g_ctx->n);
        return SUCCESS;
    }
    arg[0] = u_star;
    arg[1] = theta;
    arg[2] = variable;
    res[0] = L;
    res[1] = grad_u_L;
    return call_casadi(&g_ctx->loss_grad, arg, res) == 0 ? SUCCESS : FAILURE;
}

static int dynamic_vjp(
    const double* v,
    const double* u_star,
    const double* theta,
    const double* variable,
    double* grad_theta
)
{
    const double* arg[4];
    double* res[1];
    arg[0] = v;
    arg[1] = u_star;
    arg[2] = theta;
    arg[3] = variable;
    res[0] = grad_theta;
    return call_casadi(&g_ctx->vjp, arg, res) == 0 ? SUCCESS : FAILURE;
}

static int dynamic_constraint(
    const double* input,
    const double* theta,
    const double* variable,
    double* constraint
)
{
    const double* arg[3];
    double* res[1];
    arg[0] = input;
    arg[1] = theta;
    arg[2] = variable;
    res[0] = constraint;
    return call_casadi(&g_ctx->constraint, arg, res) == 0 ? SUCCESS : FAILURE;
}

static int dynamic_constraint_jtprod(
    const double* input,
    const double* theta,
    const double* variable,
    const double* multiplier,
    double* output_gradient
)
{
    const double* arg[4];
    double* res[1];
    arg[0] = input;
    arg[1] = theta;
    arg[2] = variable;
    arg[3] = multiplier;
    res[0] = output_gradient;
    return call_casadi(&g_ctx->constraint_jtprod, arg, res) == 0 ? SUCCESS : FAILURE;
}

static void projection_data(
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* projected_multiplier,
    double* active_penalty
)
{
    unsigned int i;
    double* c_value = (double*)mxCalloc(g_ctx->ncon > 0 ? g_ctx->ncon : 1, sizeof(double));
    (void)dynamic_constraint(input, theta, variable, c_value);
    for (i = 0; i < g_ctx->ncon; ++i) {
        double penalty = penalties[i] * g_constraint_penalty_scale;
        double shifted;
        double projected;
        int equality;
        int active;
        if (g_constraint_penalty_max > 0.0 && penalty > g_constraint_penalty_max) {
            penalty = g_constraint_penalty_max;
        }
        shifted = c_value[i] + multipliers[i] / penalty;
        projected = shifted;
        if (projected < constraint_lower[i]) projected = constraint_lower[i];
        if (projected > constraint_upper[i]) projected = constraint_upper[i];
        equality = fabs(constraint_upper[i] - constraint_lower[i]) <= 1e-8;
        active =
            shifted <= constraint_lower[i] + 1e-8 ||
            shifted >= constraint_upper[i] - 1e-8 ||
            equality;
        projected_multiplier[i] = penalty * (shifted - projected);
        active_penalty[i] = active ? penalty : 0.0;
    }
    mxFree(c_value);
}

static int dynamic_constraint_hvp(
    const double* v,
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* Hv
)
{
    unsigned int i;
    double* projected_multiplier;
    double* active_penalty;
    double* jv;
    double* curvature;
    double* gn_weight;
    double* gauss_newton;
    const double* arg_jv[4];
    const double* arg_weighted[5];
    double* res[1];
    int ok = SUCCESS;

    projected_multiplier = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    active_penalty = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    jv = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    curvature = (double*)mxCalloc(g_ctx->n, sizeof(double));
    gn_weight = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    gauss_newton = (double*)mxCalloc(g_ctx->n, sizeof(double));

    projection_data(input, theta, variable, multipliers, penalties,
                    constraint_lower, constraint_upper,
                    projected_multiplier, active_penalty);

    arg_jv[0] = v;
    arg_jv[1] = input;
    arg_jv[2] = theta;
    arg_jv[3] = variable;
    res[0] = jv;
    if (call_casadi(&g_ctx->constraint_jv, arg_jv, res) != 0) ok = FAILURE;

    if (ok == SUCCESS) {
        arg_weighted[0] = v;
        arg_weighted[1] = input;
        arg_weighted[2] = theta;
        arg_weighted[3] = variable;
        arg_weighted[4] = projected_multiplier;
        res[0] = curvature;
        if (call_casadi(&g_ctx->constraint_weighted_hvp, arg_weighted, res) != 0) ok = FAILURE;
    }

    if (ok == SUCCESS) {
        for (i = 0; i < g_ctx->ncon; ++i) gn_weight[i] = active_penalty[i] * jv[i];
        if (dynamic_constraint_jtprod(input, theta, variable, gn_weight, gauss_newton) == FAILURE) {
            ok = FAILURE;
        }
    }

    if (ok == SUCCESS) {
        for (i = 0; i < g_ctx->n; ++i) Hv[i] = curvature[i] + gauss_newton[i];
    }

    mxFree(projected_multiplier);
    mxFree(active_penalty);
    mxFree(jv);
    mxFree(curvature);
    mxFree(gn_weight);
    mxFree(gauss_newton);
    return ok;
}

static int dynamic_constraint_vjp(
    const double* v,
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* grad_theta
)
{
    unsigned int i;
    double* projected_multiplier;
    double* active_penalty;
    double* jv;
    double* curvature;
    double* gn_weight;
    double* gauss_newton;
    const double* arg_jv[4];
    const double* arg_weighted[5];
    const double* arg_theta[4];
    double* res[1];
    int ok = SUCCESS;

    projected_multiplier = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    active_penalty = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    jv = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    curvature = (double*)mxCalloc(g_ctx->ntheta, sizeof(double));
    gn_weight = (double*)mxCalloc(g_ctx->ncon, sizeof(double));
    gauss_newton = (double*)mxCalloc(g_ctx->ntheta, sizeof(double));

    projection_data(input, theta, variable, multipliers, penalties,
                    constraint_lower, constraint_upper,
                    projected_multiplier, active_penalty);

    arg_jv[0] = v;
    arg_jv[1] = input;
    arg_jv[2] = theta;
    arg_jv[3] = variable;
    res[0] = jv;
    if (call_casadi(&g_ctx->constraint_jv, arg_jv, res) != 0) ok = FAILURE;

    if (ok == SUCCESS) {
        arg_weighted[0] = v;
        arg_weighted[1] = input;
        arg_weighted[2] = theta;
        arg_weighted[3] = variable;
        arg_weighted[4] = projected_multiplier;
        res[0] = curvature;
        if (call_casadi(&g_ctx->constraint_weighted_vjp, arg_weighted, res) != 0) ok = FAILURE;
    }

    if (ok == SUCCESS) {
        for (i = 0; i < g_ctx->ncon; ++i) gn_weight[i] = active_penalty[i] * jv[i];
        arg_theta[0] = input;
        arg_theta[1] = theta;
        arg_theta[2] = variable;
        arg_theta[3] = gn_weight;
        res[0] = gauss_newton;
        if (call_casadi(&g_ctx->constraint_theta_jtprod, arg_theta, res) != 0) ok = FAILURE;
    }

    if (ok == SUCCESS) {
        for (i = 0; i < g_ctx->ntheta; ++i) grad_theta[i] = curvature[i] + gauss_newton[i];
    }

    mxFree(projected_multiplier);
    mxFree(active_penalty);
    mxFree(jv);
    mxFree(curvature);
    mxFree(gn_weight);
    mxFree(gauss_newton);
    return ok;
}

static panda_oracle build_oracle(int enable_backward)
{
    panda_oracle out;
    out.n = g_ctx->n;
    out.ntheta = g_ctx->ntheta;
    out.nvar = g_ctx->nvar;
    out.cost_gradient = dynamic_cost_gradient;
    out.proxg = dynamic_prox;
    out.jprox = enable_backward ? dynamic_jprox : 0;
    out.hvp = enable_backward ? dynamic_hvp : 0;
    out.loss_grad = enable_backward ? dynamic_loss_grad : 0;
    out.vjp = enable_backward ? dynamic_vjp : 0;
    return out;
}

static mxArray* make_output(
    const double* solution,
    const double* grad_theta,
    const double* multipliers,
    int has_grad,
    unsigned int ncon,
    unsigned int iterations,
    double final_residual,
    double penalty,
    unsigned int backward_iterations,
    double backward_residual
)
{
    const char* fields[] = {
        "solution", "grad_theta", "iterations", "final_residual",
        "penalty", "multipliers", "penalties", "backward_iterations", "backward_residual"
    };
    mxArray* out = mxCreateStructMatrix(1, 1, 9, fields);
    mxArray* sol = mxCreateDoubleMatrix(g_ctx->n, 1, mxREAL);
    memcpy(mxGetPr(sol), solution, sizeof(double) * g_ctx->n);
    mxSetField(out, 0, "solution", sol);

    if (has_grad) {
        mxArray* grad = mxCreateDoubleMatrix(g_ctx->ntheta, 1, mxREAL);
        memcpy(mxGetPr(grad), grad_theta, sizeof(double) * g_ctx->ntheta);
        mxSetField(out, 0, "grad_theta", grad);
    } else {
        mxSetField(out, 0, "grad_theta", mxCreateDoubleMatrix(0, 0, mxREAL));
    }

    mxSetField(out, 0, "iterations", mxCreateDoubleScalar((double)iterations));
    mxSetField(out, 0, "final_residual", mxCreateDoubleScalar(final_residual));
    mxSetField(out, 0, "penalty", mxCreateDoubleScalar(penalty));
    if (multipliers != 0 && ncon > 0) {
        unsigned int i;
        mxArray* multiplier_out = mxCreateDoubleMatrix(ncon, 1, mxREAL);
        mxArray* penalty_out = mxCreateDoubleMatrix(ncon, 1, mxREAL);
        memcpy(mxGetPr(multiplier_out), multipliers, sizeof(double) * ncon);
        for (i = 0; i < ncon; ++i) {
            mxGetPr(penalty_out)[i] = alm_get_penalty(i);
        }
        mxSetField(out, 0, "multipliers", multiplier_out);
        mxSetField(out, 0, "penalties", penalty_out);
    } else {
        mxSetField(out, 0, "multipliers", mxCreateDoubleMatrix(0, 0, mxREAL));
        mxSetField(out, 0, "penalties", mxCreateDoubleMatrix(0, 0, mxREAL));
    }
    mxSetField(out, 0, "backward_iterations", mxCreateDoubleScalar((double)backward_iterations));
    mxSetField(out, 0, "backward_residual", mxCreateDoubleScalar(backward_residual));
    return out;
}

static void load_context(dynamic_context* ctx, const mxArray* meta, int need_constraints, int need_backward)
{
    char* library_path;
    ctx->n = (unsigned int)get_required_scalar(meta, "n");
    ctx->ntheta = (unsigned int)get_required_scalar(meta, "ntheta");
    ctx->nvar = (unsigned int)get_required_scalar(meta, "nvar");
    ctx->ncon = (unsigned int)get_option(meta, "ncon", 0.0);
    ctx->box_lower = copy_optional_vector(meta, "box_lower", ctx->n, &ctx->has_lower);
    ctx->box_upper = copy_optional_vector(meta, "box_upper", ctx->n, &ctx->has_upper);

    library_path = get_required_string(meta, "library_path");
    ctx->library = open_library(library_path);
    mxFree(library_path);
    if (ctx->library == 0) {
        mexErrMsgIdAndTxt("lapanda:oracle", "Failed to load generated oracle library.");
    }

    load_function(ctx->library, &ctx->cost_grad, "panda_cost_grad", 1);
    if (need_backward) {
        load_function(ctx->library, &ctx->hvp, "panda_hvp", 1);
        load_function(ctx->library, &ctx->vjp, "panda_vjp", 1);
        load_function(ctx->library, &ctx->loss_grad, "panda_loss_grad", 0);
    }
    if (need_constraints) {
        load_function(ctx->library, &ctx->constraint, "panda_constraints", 1);
        load_function(ctx->library, &ctx->constraint_jtprod, "panda_constraint_jtprod", 1);
        if (need_backward) {
            load_function(ctx->library, &ctx->constraint_jv, "panda_constraint_jv", 1);
            load_function(ctx->library, &ctx->constraint_weighted_hvp, "panda_constraint_weighted_hvp", 1);
            load_function(ctx->library, &ctx->constraint_weighted_vjp, "panda_constraint_weighted_vjp", 1);
            load_function(ctx->library, &ctx->constraint_theta_jtprod, "panda_constraint_theta_jtprod", 1);
        }
    }
}

static void free_context(dynamic_context* ctx)
{
    free_function(&ctx->cost_grad);
    free_function(&ctx->hvp);
    free_function(&ctx->vjp);
    free_function(&ctx->loss_grad);
    free_function(&ctx->constraint);
    free_function(&ctx->constraint_jtprod);
    free_function(&ctx->constraint_jv);
    free_function(&ctx->constraint_weighted_hvp);
    free_function(&ctx->constraint_weighted_vjp);
    free_function(&ctx->constraint_theta_jtprod);
    if (ctx->box_lower != 0) mxFree(ctx->box_lower);
    if (ctx->box_upper != 0) mxFree(ctx->box_upper);
    close_library(ctx->library);
    memset(ctx, 0, sizeof(*ctx));
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    char mode[16];
    const mxArray* meta;
    const mxArray* options;
    const double* theta;
    const double* variable;
    double variable_dummy[1] = {0.0};
    double* solution;
    double* grad_theta;
    struct solver_parameters solver_params;
    struct backward_parameters backward_params;
    dynamic_context ctx;

    if (nrhs < 5 || !mxIsChar(prhs[0]) || !mxIsStruct(prhs[1])) {
        mexErrMsgIdAndTxt(
            "lapanda:usage",
            "Usage: result = LAPANDA_mex('panda'|'alm', meta, x0, theta, variable, ...)");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("lapanda:usage", "This MEX function returns one result struct.");
    }
    if (mxGetString(prhs[0], mode, sizeof(mode)) != 0) {
        mexErrMsgIdAndTxt("lapanda:badInput", "Mode string is too long.");
    }

    meta = prhs[1];
    options = 0;
    memset(&ctx, 0, sizeof(ctx));

    fill_backward_params(&backward_params, 0);
    if (strcmp(mode, "panda") == 0) {
        if (nrhs >= 6) options = prhs[5];
    } else if (strcmp(mode, "alm") == 0) {
        if (nrhs >= 8) options = prhs[7];
    } else {
        mexErrMsgIdAndTxt("lapanda:badInput", "Mode must be 'panda' or 'alm'.");
    }
    fill_backward_params(&backward_params, options);

    load_context(&ctx, meta, strcmp(mode, "alm") == 0, backward_params.enable);
    g_ctx = &ctx;

    check_vector(prhs[2], ctx.n, "x0");
    check_vector(prhs[3], ctx.ntheta, "theta");
    if (ctx.nvar > 0) {
        check_vector(prhs[4], ctx.nvar, "variable");
        variable = mxGetPr(prhs[4]);
    } else {
        variable = variable_dummy;
    }

    theta = mxGetPr(prhs[3]);
    solution = (double*)mxCalloc(ctx.n > 0 ? ctx.n : 1, sizeof(double));
    grad_theta = (double*)mxCalloc(ctx.ntheta > 0 ? ctx.ntheta : 1, sizeof(double));
    memcpy(solution, mxGetPr(prhs[2]), sizeof(double) * ctx.n);
    fill_solver_params(&solver_params, options);

    if (strcmp(mode, "panda") == 0) {
        struct optimizer_problem problem;
        optimizer_solve_info forward_info;
        optimizer_solve_info backward_info;
        int status;

        problem.oracle = build_oracle(backward_params.enable);
        problem.solver_params = solver_params;
        problem.backward_params = backward_params;
        problem.trace = 0;
        problem.trace_context = 0;
        if (optimizer_init(&problem) == FAILURE) {
            free_context(&ctx);
            mexErrMsgIdAndTxt("lapanda:solver", "optimizer_init failed.");
        }
        status = solve_problem(solution, theta, variable);
        (void)optimizer_get_forward_info(&forward_info);
        if (status == FAILURE && forward_info.iterations == 0) {
            optimizer_cleanup();
            free_context(&ctx);
            mexErrMsgIdAndTxt("lapanda:solver", "solve_problem failed.");
        }
        backward_info.iterations = 0;
        backward_info.final_residual = 0.0;
        if (backward_params.enable) {
            if (solve_backward(solution, grad_theta) == FAILURE) {
                optimizer_cleanup();
                free_context(&ctx);
                mexErrMsgIdAndTxt("lapanda:solver", "solve_backward failed.");
            }
            (void)optimizer_get_backward_info(&backward_info);
        }
        optimizer_cleanup();
        plhs[0] = make_output(
            solution, grad_theta, 0, backward_params.enable, 0,
            forward_info.iterations, forward_info.final_residual,
            0.0, backward_info.iterations, backward_info.final_residual);
    } else {
        alm_problem problem;
        alm_parameters alm_params;
        alm_info info;
        optimizer_solve_info backward_info;
        const double* constraint_lower;
        const double* constraint_upper;
        double* multipliers;
        double* penalty0;
        int status;

        if (ctx.ncon == 0) {
            free_context(&ctx);
            mexErrMsgIdAndTxt("lapanda:usage", "This oracle has no ALM constraints.");
        }
        if (nrhs < 7) {
            free_context(&ctx);
            mexErrMsgIdAndTxt(
                "lapanda:usage",
                "ALM usage: result = LAPANDA_mex('alm', meta, x0, theta, variable, lower, upper, options)");
        }
        check_vector(prhs[5], ctx.ncon, "constraint_lower");
        check_vector(prhs[6], ctx.ncon, "constraint_upper");
        constraint_lower = mxGetPr(prhs[5]);
        constraint_upper = mxGetPr(prhs[6]);
        multipliers = copy_option_vector(options, "alm_multiplier0", ctx.ncon);
        if (multipliers == 0) {
            multipliers = (double*)mxCalloc(ctx.ncon, sizeof(double));
        }
        penalty0 = copy_option_vector(options, "alm_penalty0", ctx.ncon);

        fill_alm_params(&alm_params, options);
        problem.oracle = build_oracle(backward_params.enable);
        problem.ncon = ctx.ncon;
        problem.constraint_lower = constraint_lower;
        problem.constraint_upper = constraint_upper;
        problem.constraint = dynamic_constraint;
        problem.constraint_jtprod = dynamic_constraint_jtprod;
        problem.constraint_hvp = backward_params.enable ? dynamic_constraint_hvp : 0;
        problem.constraint_vjp = backward_params.enable ? dynamic_constraint_vjp : 0;
        problem.inner_solver_params = solver_params;
        problem.backward_params = backward_params;

        backward_info.iterations = 0;
        backward_info.final_residual = 0.0;
        if (backward_params.enable) {
            status = alm_solve_with_backward_and_penalty0(
                &problem, &alm_params, solution, multipliers, penalty0, theta, variable,
                &info, grad_theta, &backward_info);
        } else {
            status = alm_solve_with_penalty0(
                &problem, &alm_params, solution, multipliers, penalty0, theta, variable, &info);
        }
        if (penalty0 != 0) mxFree(penalty0);
        if (status < 0 || (status == FAILURE && info.iterations == 0)) {
            mxFree(multipliers);
            free_context(&ctx);
            mexErrMsgIdAndTxt("lapanda:solver", "alm_solve failed.");
        }
        plhs[0] = make_output(
            solution, grad_theta, multipliers, backward_params.enable, ctx.ncon,
            info.iterations, info.final_residual, info.penalty,
            backward_info.iterations, backward_info.final_residual);
        mxFree(multipliers);
    }

    mxFree(solution);
    mxFree(grad_theta);
    free_context(&ctx);
    g_ctx = 0;
    g_constraint_penalty_scale = 1.0;
    g_constraint_penalty_max = 0.0;
}
