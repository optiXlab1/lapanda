#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

extern "C" {
#include "../../include/optimizer.h"
#include "../../alm/alm.h"
#include "../../panda/panda_linear_solver.h"
}

namespace py = pybind11;

namespace {

struct SolverOptions {
    unsigned int max_iterations = 600;
    double tolerance = 1e-4;
    unsigned int buffer_size = 10;
    unsigned int max_stable_iter = 0;
    bool verbose = false;
};

struct BackwardOptions {
    bool enable = false;
    double tolerance = 1e-4;
    unsigned int max_iterations = 50;
    unsigned int restart = 40;
    std::string linear_solver = "cg";
    double constraint_penalty_scale = 1.0;
    double constraint_penalty_max = 0.0;
};

struct AlmOptions {
    unsigned int max_iterations = 20;
    double tolerance = 1e-4;
    double initial_penalty = 10.0;
    double penalty_update_factor = 10.0;
    double max_penalty = 0.0;
    double sufficient_decrease_factor = 0.25;
    bool verbose = false;
    bool warm_start_inner = true;
};

struct SolveInfo {
    unsigned int iterations = 0;
    double final_residual = 0.0;
};

struct AlmSolveInfo {
    unsigned int iterations = 0;
    double final_residual = 0.0;
    double penalty = 0.0;
};

struct PythonOracle {
    unsigned int n = 0;
    unsigned int ntheta = 0;
    unsigned int nvar = 0;
    unsigned int ncon = 0;
    py::object cost_gradient;
    py::object prox;
    py::object jprox;
    py::object hvp;
    py::object loss_grad;
    py::object vjp;
    py::object constraint;
    py::object constraint_jtprod;
    py::object constraint_hvp;
    py::object constraint_vjp;
};

static PythonOracle* g_active_oracle = nullptr;
static std::string g_callback_error;
static double g_constraint_penalty_scale = 1.0;
static double g_constraint_penalty_max = 0.0;

using casadi_int = long long;
using casadi_eval_fun = int (*)(const double**, double**, casadi_int*, double*, int);
using casadi_work_fun = int (*)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);

class SharedLibrary {
public:
    explicit SharedLibrary(const std::string& path)
    {
#ifdef _WIN32
        handle_ = reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (handle_ == nullptr) {
            throw std::runtime_error("failed to load generated oracle library: " + path);
        }
    }

    ~SharedLibrary()
    {
        if (handle_ != nullptr) {
#ifdef _WIN32
            FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
        }
    }

    void* symbol(const std::string& name, bool required = true) const
    {
#ifdef _WIN32
        void* ptr = reinterpret_cast<void*>(
            GetProcAddress(reinterpret_cast<HMODULE>(handle_), name.c_str()));
#else
        void* ptr = dlsym(handle_, name.c_str());
#endif
        if (ptr == nullptr && required) {
            throw std::runtime_error("generated oracle is missing symbol: " + name);
        }
        return ptr;
    }

private:
    void* handle_ = nullptr;
};

struct CasadiFunction {
    casadi_eval_fun eval = nullptr;
    casadi_work_fun work = nullptr;
    std::vector<casadi_int> iw;
    std::vector<double> w;

    void load(const SharedLibrary& lib, const std::string& name, bool required = true)
    {
        eval = reinterpret_cast<casadi_eval_fun>(lib.symbol(name, required));
        if (eval == nullptr) {
            return;
        }
        work = reinterpret_cast<casadi_work_fun>(lib.symbol(name + "_work", false));
        if (work != nullptr) {
            casadi_int sz_arg = 0;
            casadi_int sz_res = 0;
            casadi_int sz_iw = 0;
            casadi_int sz_w = 0;
            if (work(&sz_arg, &sz_res, &sz_iw, &sz_w) != 0) {
                throw std::runtime_error("failed to query CasADi work sizes for " + name);
            }
            iw.assign(static_cast<size_t>(sz_iw), 0);
            w.assign(static_cast<size_t>(sz_w), 0.0);
        }
    }

    int operator()(const double** arg, double** res)
    {
        if (eval == nullptr) {
            return FAILURE;
        }
        return eval(arg, res, iw.empty() ? nullptr : iw.data(), w.empty() ? nullptr : w.data(), 0);
    }
};

struct CompiledOracleContext {
    std::unique_ptr<SharedLibrary> library;
    CasadiFunction cost_grad;
    CasadiFunction hvp;
    CasadiFunction vjp;
    CasadiFunction loss_grad;
    CasadiFunction constraint;
    CasadiFunction constraint_jtprod;
    CasadiFunction constraint_jv;
    CasadiFunction constraint_weighted_hvp;
    CasadiFunction constraint_weighted_vjp;
    CasadiFunction constraint_theta_jtprod;
    unsigned int n = 0;
    unsigned int ntheta = 0;
    unsigned int nvar = 0;
    unsigned int ncon = 0;
    std::vector<double> box_lower;
    std::vector<double> box_upper;
    bool has_lower = false;
    bool has_upper = false;
};

static CompiledOracleContext* g_compiled_oracle = nullptr;

py::array_t<double> array_from_ptr(const double* data, py::ssize_t size)
{
    py::array_t<double> out(size);
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < size; ++i) {
        buf(i) = data[i];
    }
    return out;
}

py::array_t<double> array_from_vector(const std::vector<double>& values)
{
    py::array_t<double> out(values.size());
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        buf(i) = values[static_cast<size_t>(i)];
    }
    return out;
}

panda_backward_solver_type backward_solver_from_name(const std::string& name)
{
    if (name == "auto") {
        return PANDA_BACKWARD_SOLVER_AUTO;
    }
    if (name == "minres") {
        return PANDA_BACKWARD_SOLVER_MINRES;
    }
    if (name == "gmres") {
        return PANDA_BACKWARD_SOLVER_GMRES;
    }
    if (name == "cg") {
        return PANDA_BACKWARD_SOLVER_CG;
    }
    throw std::runtime_error("linear_solver must be one of: auto, minres, gmres, cg");
}

const char* backward_solver_name(panda_backward_solver_type solver)
{
    if (solver == PANDA_BACKWARD_SOLVER_CG) return "cg";
    if (solver == PANDA_BACKWARD_SOLVER_MINRES) return "minres";
    if (solver == PANDA_BACKWARD_SOLVER_GMRES) return "gmres";
    return "auto";
}

struct DenseOperator {
    const double* data;
    unsigned int n;
};

int dense_matvec(const real_t* x, real_t* y, void* user_data)
{
    DenseOperator* op = static_cast<DenseOperator*>(user_data);
    for (unsigned int row = 0; row < op->n; ++row) {
        double value = 0.0;
        const double* row_data = op->data + static_cast<size_t>(row) * op->n;
        for (unsigned int col = 0; col < op->n; ++col) {
            value += row_data[col] * x[col];
        }
        y[row] = value;
    }
    return SUCCESS;
}

py::dict solve_dense_minres(
    py::array_t<double, py::array::c_style | py::array::forcecast> matrix,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
    double tolerance,
    unsigned int max_iterations)
{
    py::buffer_info matrix_info = matrix.request();
    py::buffer_info rhs_info = rhs.request();
    const size_t n = static_cast<size_t>(rhs_info.size);

    if (matrix_info.ndim != 2 ||
        matrix_info.shape[0] != static_cast<py::ssize_t>(n) ||
        matrix_info.shape[1] != static_cast<py::ssize_t>(n)) {
        throw std::runtime_error("dense matrix must be square and match rhs");
    }
    if (n > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error("dense system is too large for the native MINRES interface");
    }

    DenseOperator op{
        static_cast<const double*>(matrix_info.ptr),
        static_cast<unsigned int>(n),
    };
    std::vector<double> solution(n, 0.0);
    double relative_residual = 0.0;
    unsigned int iterations = 0;
    int status = FAILURE;
    {
        py::gil_scoped_release release;
        status = panda_minres_solve(
            dense_matvec,
            &op,
            static_cast<const double*>(rhs_info.ptr),
            solution.data(),
            static_cast<unsigned int>(n),
            tolerance,
            max_iterations,
            &relative_residual,
            &iterations);
    }

    py::dict result;
    result["solution"] = array_from_vector(solution);
    result["status"] = status;
    result["relative_residual"] = relative_residual;
    result["iterations"] = iterations;
    result["peak_workspace_bytes"] =
        py::int_(panda_linear_solver_get_last_peak_workspace_bytes());
    return result;
}

std::vector<double> scaled_constraint_penalties(const double* penalties, unsigned int ncon)
{
    std::vector<double> scaled(ncon);
    for (unsigned int i = 0; i < ncon; ++i) {
        scaled[i] = penalties[i] * g_constraint_penalty_scale;
        if (g_constraint_penalty_max > 0.0) {
            scaled[i] = std::min(scaled[i], g_constraint_penalty_max);
        }
    }
    return scaled;
}

std::vector<double> vector_from_object(const py::object& obj)
{
    py::module_ np = py::module_::import("numpy");
    py::array array = np.attr("asarray")(obj, py::arg("dtype") = np.attr("float64"));
    py::buffer_info info = array.request();
    if (info.ndim == 0) {
        return { *static_cast<double*>(info.ptr) };
    }

    size_t count = 1;
    for (auto dim : info.shape) {
        count *= static_cast<size_t>(dim);
    }

    const double* ptr = static_cast<const double*>(info.ptr);
    return std::vector<double>(ptr, ptr + count);
}

double scalar_from_object(const py::object& obj)
{
    return py::cast<double>(obj);
}

void copy_object_to_ptr(const py::object& obj, double* dst, size_t expected)
{
    std::vector<double> values = vector_from_object(obj);
    if (values.size() != expected) {
        throw std::runtime_error("oracle callback returned an array with an unexpected size");
    }
    std::copy(values.begin(), values.end(), dst);
}

py::object tuple_item(const py::object& obj, py::ssize_t index)
{
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        return seq[index];
    }
    if (index == 0) {
        return obj;
    }
    throw std::runtime_error("oracle callback must return a tuple/list for this operation");
}

void save_callback_error()
{
    try {
        throw;
    } catch (const std::exception& e) {
        g_callback_error = e.what();
    } catch (...) {
        g_callback_error = "unknown Python callback error";
    }
}

double py_cost_gradient(
    const double* input,
    const double* theta,
    const double* variable,
    double* output_gradient)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->cost_gradient(
            array_from_ptr(input, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        double cost = scalar_from_object(tuple_item(result, 0));
        copy_object_to_ptr(tuple_item(result, 1), output_gradient, g_active_oracle->n);
        return cost;
    } catch (...) {
        save_callback_error();
        std::fill(output_gradient, output_gradient + g_active_oracle->n, 0.0);
        return 0.0;
    }
}

double py_prox(
    double* input,
    double gamma,
    const double* theta,
    const double* variable)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->prox(
            array_from_ptr(input, g_active_oracle->n),
            gamma,
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        copy_object_to_ptr(tuple_item(result, 0), input, g_active_oracle->n);
        return scalar_from_object(tuple_item(result, 1));
    } catch (...) {
        save_callback_error();
        return 0.0;
    }
}

int py_jprox(
    const double* u_star,
    double gamma,
    const double* theta,
    const double* variable,
    double* J)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->jprox(
            array_from_ptr(u_star, g_active_oracle->n),
            gamma,
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        copy_object_to_ptr(result, J, g_active_oracle->n);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_hvp(
    const double* v,
    const double* u_star,
    const double* theta,
    const double* variable,
    double* Hv)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->hvp(
            array_from_ptr(v, g_active_oracle->n),
            array_from_ptr(u_star, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        copy_object_to_ptr(result, Hv, g_active_oracle->n);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_loss_grad(
    const double* u_star,
    const double* theta,
    const double* variable,
    double* L,
    double* grad_u_L)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->loss_grad(
            array_from_ptr(u_star, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        *L = scalar_from_object(tuple_item(result, 0));
        copy_object_to_ptr(tuple_item(result, 1), grad_u_L, g_active_oracle->n);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_vjp(
    const double* v,
    const double* u_star,
    const double* theta,
    const double* variable,
    double* grad_theta)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->vjp(
            array_from_ptr(v, g_active_oracle->n),
            array_from_ptr(u_star, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        copy_object_to_ptr(result, grad_theta, g_active_oracle->ntheta);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_constraint(
    const double* input,
    const double* theta,
    const double* variable,
    double* constraint)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->constraint(
            array_from_ptr(input, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar));
        copy_object_to_ptr(result, constraint, g_active_oracle->ncon);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_constraint_jtprod(
    const double* input,
    const double* theta,
    const double* variable,
    const double* multiplier,
    double* output_gradient)
{
    try {
        py::gil_scoped_acquire gil;
        py::object result = g_active_oracle->constraint_jtprod(
            array_from_ptr(input, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar),
            array_from_ptr(multiplier, g_active_oracle->ncon));
        copy_object_to_ptr(result, output_gradient, g_active_oracle->n);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_constraint_hvp(
    const double* v,
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* Hv)
{
    try {
        py::gil_scoped_acquire gil;
        std::vector<double> scaled_penalties =
            scaled_constraint_penalties(penalties, g_active_oracle->ncon);
        py::object result = g_active_oracle->constraint_hvp(
            array_from_ptr(v, g_active_oracle->n),
            array_from_ptr(input, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar),
            array_from_ptr(multipliers, g_active_oracle->ncon),
            array_from_vector(scaled_penalties),
            array_from_ptr(constraint_lower, g_active_oracle->ncon),
            array_from_ptr(constraint_upper, g_active_oracle->ncon));
        copy_object_to_ptr(result, Hv, g_active_oracle->n);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

int py_constraint_vjp(
    const double* v,
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* grad_theta)
{
    try {
        py::gil_scoped_acquire gil;
        std::vector<double> scaled_penalties =
            scaled_constraint_penalties(penalties, g_active_oracle->ncon);
        py::object result = g_active_oracle->constraint_vjp(
            array_from_ptr(v, g_active_oracle->n),
            array_from_ptr(input, g_active_oracle->n),
            array_from_ptr(theta, g_active_oracle->ntheta),
            array_from_ptr(variable, g_active_oracle->nvar),
            array_from_ptr(multipliers, g_active_oracle->ncon),
            array_from_vector(scaled_penalties),
            array_from_ptr(constraint_lower, g_active_oracle->ncon),
            array_from_ptr(constraint_upper, g_active_oracle->ncon));
        copy_object_to_ptr(result, grad_theta, g_active_oracle->ntheta);
        return SUCCESS;
    } catch (...) {
        save_callback_error();
        return FAILURE;
    }
}

std::vector<double> checked_vector(py::array_t<double, py::array::c_style | py::array::forcecast> array,
                                   size_t expected,
                                   const char* name)
{
    py::buffer_info info = array.request();
    size_t count = 1;
    for (auto dim : info.shape) {
        count *= static_cast<size_t>(dim);
    }
    if (count != expected) {
        throw std::runtime_error(std::string(name) + " has unexpected size");
    }
    const double* ptr = static_cast<const double*>(info.ptr);
    return std::vector<double>(ptr, ptr + count);
}

std::vector<double> vector_from_array(py::array array)
{
    return vector_from_object(py::reinterpret_borrow<py::object>(array));
}

std::vector<double> optional_vector_from_object(const py::object& obj, size_t expected, const char* name)
{
    if (obj.is_none()) {
        return {};
    }
    std::vector<double> values = vector_from_object(obj);
    if (values.size() != expected) {
        throw std::runtime_error(std::string(name) + " has unexpected size");
    }
    return values;
}

double compiled_cost_gradient(
    const double* input,
    const double* theta,
    const double* variable,
    double* output_gradient)
{
    double value = 0.0;
    const double* arg[] = { input, theta, variable };
    double* res[] = { &value, output_gradient };
    if ((*g_compiled_oracle).cost_grad(arg, res) != 0) {
        return value;
    }
    return value;
}

double compiled_prox(real_t* input, real_t gamma, const real_t* theta, const real_t* variable)
{
    (void)gamma;
    (void)theta;
    (void)variable;
    for (unsigned int i = 0; i < g_compiled_oracle->n; ++i) {
        if (g_compiled_oracle->has_lower) {
            input[i] = std::max(input[i], g_compiled_oracle->box_lower[i]);
        }
        if (g_compiled_oracle->has_upper) {
            input[i] = std::min(input[i], g_compiled_oracle->box_upper[i]);
        }
    }
    return 0.0;
}

int compiled_jprox(
    const real_t* u_star,
    real_t gamma,
    const real_t* theta,
    const real_t* variable,
    real_t* J)
{
    (void)gamma;
    (void)theta;
    (void)variable;
    const double active_tol = 1e-10;
    for (unsigned int i = 0; i < g_compiled_oracle->n; ++i) {
        J[i] = 1.0;
        if (g_compiled_oracle->has_lower &&
            u_star[i] <= g_compiled_oracle->box_lower[i] + active_tol) {
            J[i] = 0.0;
        }
        if (g_compiled_oracle->has_upper &&
            u_star[i] >= g_compiled_oracle->box_upper[i] - active_tol) {
            J[i] = 0.0;
        }
    }
    return SUCCESS;
}

int compiled_hvp(
    const double* v,
    const double* u_star,
    const double* theta,
    const double* variable,
    double* Hv)
{
    const double* arg[] = { v, u_star, theta, variable };
    double* res[] = { Hv };
    return (*g_compiled_oracle).hvp(arg, res) == 0 ? SUCCESS : FAILURE;
}

int compiled_loss_grad(
    const double* u_star,
    const double* theta,
    const double* variable,
    double* L,
    double* grad_u_L)
{
    if (g_compiled_oracle->loss_grad.eval == nullptr) {
        *L = 0.0;
        std::fill(grad_u_L, grad_u_L + g_compiled_oracle->n, 0.0);
        return SUCCESS;
    }
    const double* arg[] = { u_star, theta, variable };
    double* res[] = { L, grad_u_L };
    return (*g_compiled_oracle).loss_grad(arg, res) == 0 ? SUCCESS : FAILURE;
}

int compiled_vjp(
    const double* v,
    const double* u_star,
    const double* theta,
    const double* variable,
    double* grad_theta)
{
    const double* arg[] = { v, u_star, theta, variable };
    double* res[] = { grad_theta };
    return (*g_compiled_oracle).vjp(arg, res) == 0 ? SUCCESS : FAILURE;
}

int compiled_constraint(
    const double* input,
    const double* theta,
    const double* variable,
    double* constraint)
{
    const double* arg[] = { input, theta, variable };
    double* res[] = { constraint };
    return (*g_compiled_oracle).constraint(arg, res) == 0 ? SUCCESS : FAILURE;
}

int compiled_constraint_jtprod(
    const double* input,
    const double* theta,
    const double* variable,
    const double* multiplier,
    double* output_gradient)
{
    const double* arg[] = { input, theta, variable, multiplier };
    double* res[] = { output_gradient };
    return (*g_compiled_oracle).constraint_jtprod(arg, res) == 0 ? SUCCESS : FAILURE;
}

void compiled_constraint_projection_data(
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    std::vector<double>& projected_multiplier,
    std::vector<double>& active_penalty)
{
    const unsigned int ncon = g_compiled_oracle->ncon;
    std::vector<double> c_value(ncon, 0.0);
    if (compiled_constraint(input, theta, variable, c_value.data()) == FAILURE) {
        throw std::runtime_error("compiled constraint evaluation failed");
    }
    std::vector<double> scaled_penalties = scaled_constraint_penalties(penalties, ncon);
    projected_multiplier.assign(ncon, 0.0);
    active_penalty.assign(ncon, 0.0);
    for (unsigned int i = 0; i < ncon; ++i) {
        const double shifted = c_value[i] + multipliers[i] / scaled_penalties[i];
        const double projected = std::min(
            std::max(shifted, constraint_lower[i]),
            constraint_upper[i]);
        const bool equality = std::abs(constraint_upper[i] - constraint_lower[i]) <= 1e-8;
        const bool active =
            shifted <= constraint_lower[i] + 1e-8 ||
            shifted >= constraint_upper[i] - 1e-8 ||
            equality;
        projected_multiplier[i] = scaled_penalties[i] * (shifted - projected);
        active_penalty[i] = active ? scaled_penalties[i] : 0.0;
    }
}

int compiled_constraint_hvp(
    const double* v,
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* Hv)
{
    try {
        const unsigned int n = g_compiled_oracle->n;
        const unsigned int ncon = g_compiled_oracle->ncon;
        std::vector<double> projected_multiplier;
        std::vector<double> active_penalty;
        std::vector<double> jv(ncon, 0.0);
        std::vector<double> curvature(n, 0.0);
        std::vector<double> gn_weight(ncon, 0.0);
        std::vector<double> gauss_newton(n, 0.0);
        compiled_constraint_projection_data(
            input,
            theta,
            variable,
            multipliers,
            penalties,
            constraint_lower,
            constraint_upper,
            projected_multiplier,
            active_penalty);
        {
            const double* arg[] = { v, input, theta, variable };
            double* res[] = { jv.data() };
            if ((*g_compiled_oracle).constraint_jv(arg, res) != 0) {
                return FAILURE;
            }
        }
        {
            const double* arg[] = { v, input, theta, variable, projected_multiplier.data() };
            double* res[] = { curvature.data() };
            if ((*g_compiled_oracle).constraint_weighted_hvp(arg, res) != 0) {
                return FAILURE;
            }
        }
        for (unsigned int i = 0; i < ncon; ++i) {
            gn_weight[i] = active_penalty[i] * jv[i];
        }
        if (compiled_constraint_jtprod(input, theta, variable, gn_weight.data(), gauss_newton.data()) == FAILURE) {
            return FAILURE;
        }
        for (unsigned int i = 0; i < n; ++i) {
            Hv[i] = curvature[i] + gauss_newton[i];
        }
        return SUCCESS;
    } catch (...) {
        return FAILURE;
    }
}

int compiled_constraint_vjp(
    const double* v,
    const double* input,
    const double* theta,
    const double* variable,
    const double* multipliers,
    const double* penalties,
    const double* constraint_lower,
    const double* constraint_upper,
    double* grad_theta)
{
    try {
        const unsigned int ntheta = g_compiled_oracle->ntheta;
        const unsigned int ncon = g_compiled_oracle->ncon;
        std::vector<double> projected_multiplier;
        std::vector<double> active_penalty;
        std::vector<double> jv(ncon, 0.0);
        std::vector<double> curvature(ntheta, 0.0);
        std::vector<double> gn_weight(ncon, 0.0);
        std::vector<double> gauss_newton(ntheta, 0.0);
        compiled_constraint_projection_data(
            input,
            theta,
            variable,
            multipliers,
            penalties,
            constraint_lower,
            constraint_upper,
            projected_multiplier,
            active_penalty);
        {
            const double* arg[] = { v, input, theta, variable };
            double* res[] = { jv.data() };
            if ((*g_compiled_oracle).constraint_jv(arg, res) != 0) {
                return FAILURE;
            }
        }
        {
            const double* arg[] = { v, input, theta, variable, projected_multiplier.data() };
            double* res[] = { curvature.data() };
            if ((*g_compiled_oracle).constraint_weighted_vjp(arg, res) != 0) {
                return FAILURE;
            }
        }
        for (unsigned int i = 0; i < ncon; ++i) {
            gn_weight[i] = active_penalty[i] * jv[i];
        }
        {
            const double* arg[] = { input, theta, variable, gn_weight.data() };
            double* res[] = { gauss_newton.data() };
            if ((*g_compiled_oracle).constraint_theta_jtprod(arg, res) != 0) {
                return FAILURE;
            }
        }
        for (unsigned int i = 0; i < ntheta; ++i) {
            grad_theta[i] = curvature[i] + gauss_newton[i];
        }
        return SUCCESS;
    } catch (...) {
        return FAILURE;
    }
}

panda_oracle build_compiled_c_oracle(const CompiledOracleContext& ctx, bool enable_backward)
{
    panda_oracle out;
    out.n = ctx.n;
    out.ntheta = ctx.ntheta;
    out.nvar = ctx.nvar;
    out.cost_gradient = compiled_cost_gradient;
    out.proxg = compiled_prox;
    out.jprox = enable_backward ? compiled_jprox : nullptr;
    out.hvp = enable_backward ? compiled_hvp : nullptr;
    out.loss_grad = enable_backward ? compiled_loss_grad : nullptr;
    out.vjp = enable_backward ? compiled_vjp : nullptr;
    return out;
}

panda_oracle build_c_oracle(PythonOracle& oracle, bool enable_backward)
{
    panda_oracle out;
    out.n = oracle.n;
    out.ntheta = oracle.ntheta;
    out.nvar = oracle.nvar;
    out.cost_gradient = py_cost_gradient;
    out.proxg = py_prox;
    out.jprox = enable_backward ? py_jprox : nullptr;
    out.hvp = enable_backward ? py_hvp : nullptr;
    out.loss_grad = enable_backward ? py_loss_grad : nullptr;
    out.vjp = enable_backward ? py_vjp : nullptr;
    return out;
}

void throw_if_callback_failed()
{
    if (!g_callback_error.empty()) {
        std::string msg = g_callback_error;
        g_callback_error.clear();
        throw std::runtime_error(msg);
    }
}

py::array_t<unsigned int> current_alm_inner_iterations()
{
    unsigned int count = alm_get_inner_iterations_count();
    py::array_t<unsigned int> out(count);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < count; ++i) {
        buf(i) = alm_get_inner_iterations(i);
    }
    return out;
}

py::array_t<double> current_alm_penalties()
{
    unsigned int count = alm_get_penalties_count();
    py::array_t<double> out(count);
    auto buf = out.mutable_unchecked<1>();
    for (unsigned int i = 0; i < count; ++i) {
        buf(i) = alm_get_penalty(i);
    }
    return out;
}

py::dict solve_panda_impl(
    PythonOracle& oracle,
    py::array_t<double, py::array::c_style | py::array::forcecast> x0,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta,
    py::array_t<double, py::array::c_style | py::array::forcecast> variable,
    const SolverOptions& solver_options,
    const BackwardOptions& backward_options)
{
    std::vector<double> solution = checked_vector(x0, oracle.n, "x0");
    std::vector<double> theta_vec = checked_vector(theta, oracle.ntheta, "theta");
    std::vector<double> variable_vec = checked_vector(variable, oracle.nvar, "variable");

    g_active_oracle = &oracle;
    g_callback_error.clear();

    optimizer_problem problem;
    problem.oracle = build_c_oracle(oracle, backward_options.enable);
    problem.solver_params.max_iterations = solver_options.max_iterations;
    problem.solver_params.tolerance = solver_options.tolerance;
    problem.solver_params.buffer_size = solver_options.buffer_size;
    problem.solver_params.max_stable_iter = solver_options.max_stable_iter;
    problem.solver_params.verbose = solver_options.verbose ? TRUE : FALSE;
    problem.backward_params.enable = backward_options.enable ? TRUE : FALSE;
    problem.backward_params.tolerance = backward_options.tolerance;
    problem.backward_params.max_iterations = backward_options.max_iterations;
    problem.backward_params.restart = backward_options.restart;
    problem.backward_params.force_solver = backward_solver_from_name(backward_options.linear_solver);
    problem.trace = NULL;
    problem.trace_context = NULL;

    if (optimizer_init(&problem) == FAILURE) {
        g_active_oracle = nullptr;
        throw_if_callback_failed();
        throw std::runtime_error("optimizer_init failed");
    }

    int status = solve_problem(solution.data(), theta_vec.data(), variable_vec.data());
    throw_if_callback_failed();
    optimizer_solve_info forward_info;
    optimizer_get_forward_info(&forward_info);
    if (status == FAILURE &&
        forward_info.iterations == 0 &&
        forward_info.final_residual == 0.0) {
        optimizer_cleanup();
        g_active_oracle = nullptr;
        throw std::runtime_error("solve_problem failed");
    }

    py::dict result;
    result["solution"] = array_from_vector(solution);
    result["iterations"] = forward_info.iterations;
    result["final_residual"] = forward_info.final_residual;
    result["gamma"] = get_gamma();

    if (backward_options.enable) {
        std::vector<double> grad_theta(oracle.ntheta, 0.0);
        if (solve_backward(solution.data(), grad_theta.data()) == FAILURE) {
            optimizer_cleanup();
            g_active_oracle = nullptr;
            throw_if_callback_failed();
            throw std::runtime_error("solve_backward failed");
        }
        optimizer_solve_info backward_info;
        optimizer_get_backward_info(&backward_info);
        result["grad_theta"] = array_from_vector(grad_theta);
        result["backward_iterations"] = backward_info.iterations;
        result["backward_residual"] = backward_info.final_residual;
        result["backward_peak_workspace_bytes"] =
            py::int_(panda_backward_get_last_peak_workspace_bytes());
        result["backward_solver_used"] =
            backward_solver_name(panda_backward_get_last_solver_used());
        result["backward_fallback_used"] =
            panda_backward_get_last_fallback_used() != FALSE;
    }

    optimizer_cleanup();
    g_active_oracle = nullptr;
    return result;
}

py::dict solve_panda_compiled_impl(
    const std::string& library_path,
    unsigned int n,
    unsigned int ntheta,
    unsigned int nvar,
    py::object box_lower,
    py::object box_upper,
    py::array_t<double, py::array::c_style | py::array::forcecast> x0,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta,
    py::array_t<double, py::array::c_style | py::array::forcecast> variable,
    const SolverOptions& solver_options,
    const BackwardOptions& backward_options)
{
    std::vector<double> solution = checked_vector(x0, n, "x0");
    std::vector<double> theta_vec = checked_vector(theta, ntheta, "theta");
    std::vector<double> variable_vec = checked_vector(variable, nvar, "variable");

    CompiledOracleContext ctx;
    ctx.n = n;
    ctx.ntheta = ntheta;
    ctx.nvar = nvar;
    ctx.box_lower = optional_vector_from_object(box_lower, n, "box_lower");
    ctx.box_upper = optional_vector_from_object(box_upper, n, "box_upper");
    ctx.has_lower = !ctx.box_lower.empty();
    ctx.has_upper = !ctx.box_upper.empty();
    ctx.library.reset(new SharedLibrary(library_path));
    ctx.cost_grad.load(*ctx.library, "panda_cost_grad");
    if (backward_options.enable) {
        ctx.hvp.load(*ctx.library, "panda_hvp");
        ctx.vjp.load(*ctx.library, "panda_vjp");
        ctx.loss_grad.load(*ctx.library, "panda_loss_grad", false);
    }

    g_compiled_oracle = &ctx;

    optimizer_problem problem;
    problem.oracle = build_compiled_c_oracle(ctx, backward_options.enable);
    problem.solver_params.max_iterations = solver_options.max_iterations;
    problem.solver_params.tolerance = solver_options.tolerance;
    problem.solver_params.buffer_size = solver_options.buffer_size;
    problem.solver_params.max_stable_iter = solver_options.max_stable_iter;
    problem.solver_params.verbose = solver_options.verbose ? TRUE : FALSE;
    problem.backward_params.enable = backward_options.enable ? TRUE : FALSE;
    problem.backward_params.tolerance = backward_options.tolerance;
    problem.backward_params.max_iterations = backward_options.max_iterations;
    problem.backward_params.restart = backward_options.restart;
    problem.backward_params.force_solver = backward_solver_from_name(backward_options.linear_solver);
    problem.trace = NULL;
    problem.trace_context = NULL;

    if (optimizer_init(&problem) == FAILURE) {
        g_compiled_oracle = nullptr;
        throw std::runtime_error("optimizer_init failed");
    }

    int status = solve_problem(solution.data(), theta_vec.data(), variable_vec.data());
    optimizer_solve_info forward_info;
    optimizer_get_forward_info(&forward_info);
    if (status == FAILURE &&
        forward_info.iterations == 0 &&
        forward_info.final_residual == 0.0) {
        optimizer_cleanup();
        g_compiled_oracle = nullptr;
        throw std::runtime_error("solve_problem failed");
    }

    py::dict result;
    result["solution"] = array_from_vector(solution);
    result["iterations"] = forward_info.iterations;
    result["final_residual"] = forward_info.final_residual;
    result["gamma"] = get_gamma();

    if (backward_options.enable) {
        std::vector<double> grad_theta(ntheta, 0.0);
        if (solve_backward(solution.data(), grad_theta.data()) == FAILURE) {
            optimizer_cleanup();
            g_compiled_oracle = nullptr;
            throw std::runtime_error("solve_backward failed");
        }
        optimizer_solve_info backward_info;
        optimizer_get_backward_info(&backward_info);
        result["grad_theta"] = array_from_vector(grad_theta);
        result["backward_iterations"] = backward_info.iterations;
        result["backward_residual"] = backward_info.final_residual;
        result["backward_peak_workspace_bytes"] =
            py::int_(panda_backward_get_last_peak_workspace_bytes());
        result["backward_solver_used"] =
            backward_solver_name(panda_backward_get_last_solver_used());
        result["backward_fallback_used"] =
            panda_backward_get_last_fallback_used() != FALSE;
    }

    optimizer_cleanup();
    g_compiled_oracle = nullptr;
    return result;
}

py::dict solve_alm_impl(
    PythonOracle& oracle,
    py::array_t<double, py::array::c_style | py::array::forcecast> x0,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta,
    py::array_t<double, py::array::c_style | py::array::forcecast> variable,
    py::array_t<double, py::array::c_style | py::array::forcecast> constraint_lower,
    py::array_t<double, py::array::c_style | py::array::forcecast> constraint_upper,
    const SolverOptions& inner_solver_options,
    const AlmOptions& alm_options,
    const BackwardOptions& backward_options,
    py::object multiplier0,
    py::object penalty0)
{
    std::vector<double> solution = checked_vector(x0, oracle.n, "x0");
    std::vector<double> theta_vec = checked_vector(theta, oracle.ntheta, "theta");
    std::vector<double> variable_vec = checked_vector(variable, oracle.nvar, "variable");

    std::vector<double> lower = vector_from_array(constraint_lower);
    std::vector<double> upper = vector_from_array(constraint_upper);
    if (lower.empty() || lower.size() != upper.size()) {
        throw std::runtime_error("constraint bounds must be non-empty arrays with the same size");
    }

    g_active_oracle = &oracle;
    g_constraint_penalty_scale = backward_options.constraint_penalty_scale;
    g_constraint_penalty_max = backward_options.constraint_penalty_max;
    oracle.ncon = static_cast<unsigned int>(lower.size());
    g_callback_error.clear();

    alm_problem problem;
    problem.oracle = build_c_oracle(oracle, backward_options.enable);
    problem.ncon = static_cast<unsigned int>(lower.size());
    problem.constraint_lower = lower.data();
    problem.constraint_upper = upper.data();
    problem.constraint = py_constraint;
    problem.constraint_jtprod = py_constraint_jtprod;
    problem.constraint_hvp = backward_options.enable ? py_constraint_hvp : nullptr;
    problem.constraint_vjp = backward_options.enable ? py_constraint_vjp : nullptr;
    problem.inner_solver_params.max_iterations = inner_solver_options.max_iterations;
    problem.inner_solver_params.tolerance = inner_solver_options.tolerance;
    problem.inner_solver_params.buffer_size = inner_solver_options.buffer_size;
    problem.inner_solver_params.max_stable_iter = inner_solver_options.max_stable_iter;
    problem.inner_solver_params.verbose = inner_solver_options.verbose ? TRUE : FALSE;
    problem.backward_params.enable = backward_options.enable ? TRUE : FALSE;
    problem.backward_params.tolerance = backward_options.tolerance;
    problem.backward_params.max_iterations = backward_options.max_iterations;
    problem.backward_params.restart = backward_options.restart;
    problem.backward_params.force_solver = backward_solver_from_name(backward_options.linear_solver);

    alm_parameters params;
    params.max_iterations = alm_options.max_iterations;
    params.tolerance = alm_options.tolerance;
    params.initial_penalty = alm_options.initial_penalty;
    params.penalty_update_factor = alm_options.penalty_update_factor;
    params.max_penalty = alm_options.max_penalty;
    params.sufficient_decrease_factor = alm_options.sufficient_decrease_factor;
    params.verbose = alm_options.verbose ? TRUE : FALSE;
    params.warm_start_inner = alm_options.warm_start_inner ? TRUE : FALSE;

    std::vector<double> multipliers =
        optional_vector_from_object(multiplier0, lower.size(), "multiplier0");
    if (multipliers.empty()) {
        multipliers.assign(lower.size(), 0.0);
    }
    std::vector<double> penalties0 =
        optional_vector_from_object(penalty0, lower.size(), "penalty0");
    std::vector<double> grad_theta(oracle.ntheta, 0.0);
    alm_info info;
    optimizer_solve_info backward_info;
    int status;

    info.iterations = 0;
    info.final_residual = 0.0;
    info.penalty = 0.0;
    backward_info.iterations = 0;
    backward_info.final_residual = 0.0;

    if (backward_options.enable) {
        status = alm_solve_with_backward_and_penalty0(
            &problem,
            &params,
            solution.data(),
            multipliers.data(),
            penalties0.empty() ? nullptr : penalties0.data(),
            theta_vec.data(),
            variable_vec.data(),
            &info,
            grad_theta.data(),
            &backward_info);
    } else {
        status = alm_solve_with_penalty0(
        &problem,
        &params,
        solution.data(),
        multipliers.data(),
        penalties0.empty() ? nullptr : penalties0.data(),
        theta_vec.data(),
        variable_vec.data(),
        &info);
    }
    throw_if_callback_failed();
    g_active_oracle = nullptr;
    g_constraint_penalty_scale = 1.0;
    g_constraint_penalty_max = 0.0;

    if (status == FAILURE && info.iterations == 0) {
        throw std::runtime_error("alm_solve failed");
    }
    if (status < 0) {
        throw std::runtime_error("alm_solve backward failed");
    }

    py::dict result;
    result["solution"] = array_from_vector(solution);
    result["multipliers"] = array_from_vector(multipliers);
    result["iterations"] = info.iterations;
    result["final_residual"] = info.final_residual;
    result["penalty"] = info.penalty;
    result["forward_time_sec"] = info.forward_time_sec;
    result["backward_time_sec"] = info.backward_time_sec;
    result["inner_iterations"] = current_alm_inner_iterations();
    result["penalties"] = current_alm_penalties();
    if (backward_options.enable) {
        result["grad_theta"] = array_from_vector(grad_theta);
        result["backward_iterations"] = backward_info.iterations;
        result["backward_residual"] = backward_info.final_residual;
        result["backward_peak_workspace_bytes"] =
            py::int_(panda_backward_get_last_peak_workspace_bytes());
        result["backward_solver_used"] =
            backward_solver_name(panda_backward_get_last_solver_used());
        result["backward_fallback_used"] =
            panda_backward_get_last_fallback_used() != FALSE;
    }
    return result;
}

py::dict solve_alm_compiled_impl(
    const std::string& library_path,
    unsigned int n,
    unsigned int ntheta,
    unsigned int nvar,
    unsigned int ncon,
    py::object box_lower,
    py::object box_upper,
    py::array_t<double, py::array::c_style | py::array::forcecast> x0,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta,
    py::array_t<double, py::array::c_style | py::array::forcecast> variable,
    py::array_t<double, py::array::c_style | py::array::forcecast> constraint_lower,
    py::array_t<double, py::array::c_style | py::array::forcecast> constraint_upper,
    const SolverOptions& inner_solver_options,
    const AlmOptions& alm_options,
    const BackwardOptions& backward_options,
    py::object multiplier0,
    py::object penalty0)
{
    std::vector<double> solution = checked_vector(x0, n, "x0");
    std::vector<double> theta_vec = checked_vector(theta, ntheta, "theta");
    std::vector<double> variable_vec = checked_vector(variable, nvar, "variable");
    std::vector<double> lower = checked_vector(constraint_lower, ncon, "constraint_lower");
    std::vector<double> upper = checked_vector(constraint_upper, ncon, "constraint_upper");

    CompiledOracleContext ctx;
    ctx.n = n;
    ctx.ntheta = ntheta;
    ctx.nvar = nvar;
    ctx.ncon = ncon;
    ctx.box_lower = optional_vector_from_object(box_lower, n, "box_lower");
    ctx.box_upper = optional_vector_from_object(box_upper, n, "box_upper");
    ctx.has_lower = !ctx.box_lower.empty();
    ctx.has_upper = !ctx.box_upper.empty();
    ctx.library.reset(new SharedLibrary(library_path));
    ctx.cost_grad.load(*ctx.library, "panda_cost_grad");
    ctx.constraint.load(*ctx.library, "panda_constraints");
    ctx.constraint_jtprod.load(*ctx.library, "panda_constraint_jtprod");
    if (backward_options.enable) {
        ctx.hvp.load(*ctx.library, "panda_hvp");
        ctx.vjp.load(*ctx.library, "panda_vjp");
        ctx.loss_grad.load(*ctx.library, "panda_loss_grad", false);
        ctx.constraint_jv.load(*ctx.library, "panda_constraint_jv");
        ctx.constraint_weighted_hvp.load(*ctx.library, "panda_constraint_weighted_hvp");
        ctx.constraint_weighted_vjp.load(*ctx.library, "panda_constraint_weighted_vjp");
        ctx.constraint_theta_jtprod.load(*ctx.library, "panda_constraint_theta_jtprod");
    }

    g_compiled_oracle = &ctx;
    g_constraint_penalty_scale = backward_options.constraint_penalty_scale;
    g_constraint_penalty_max = backward_options.constraint_penalty_max;

    alm_problem problem;
    problem.oracle = build_compiled_c_oracle(ctx, backward_options.enable);
    problem.ncon = ncon;
    problem.constraint_lower = lower.data();
    problem.constraint_upper = upper.data();
    problem.constraint = compiled_constraint;
    problem.constraint_jtprod = compiled_constraint_jtprod;
    problem.constraint_hvp = backward_options.enable ? compiled_constraint_hvp : nullptr;
    problem.constraint_vjp = backward_options.enable ? compiled_constraint_vjp : nullptr;
    problem.inner_solver_params.max_iterations = inner_solver_options.max_iterations;
    problem.inner_solver_params.tolerance = inner_solver_options.tolerance;
    problem.inner_solver_params.buffer_size = inner_solver_options.buffer_size;
    problem.inner_solver_params.max_stable_iter = inner_solver_options.max_stable_iter;
    problem.inner_solver_params.verbose = inner_solver_options.verbose ? TRUE : FALSE;
    problem.backward_params.enable = backward_options.enable ? TRUE : FALSE;
    problem.backward_params.tolerance = backward_options.tolerance;
    problem.backward_params.max_iterations = backward_options.max_iterations;
    problem.backward_params.restart = backward_options.restart;
    problem.backward_params.force_solver = backward_solver_from_name(backward_options.linear_solver);

    alm_parameters params;
    params.max_iterations = alm_options.max_iterations;
    params.tolerance = alm_options.tolerance;
    params.initial_penalty = alm_options.initial_penalty;
    params.penalty_update_factor = alm_options.penalty_update_factor;
    params.max_penalty = alm_options.max_penalty;
    params.sufficient_decrease_factor = alm_options.sufficient_decrease_factor;
    params.verbose = alm_options.verbose ? TRUE : FALSE;
    params.warm_start_inner = alm_options.warm_start_inner ? TRUE : FALSE;

    std::vector<double> multipliers =
        optional_vector_from_object(multiplier0, ncon, "multiplier0");
    if (multipliers.empty()) {
        multipliers.assign(ncon, 0.0);
    }
    std::vector<double> penalties0 =
        optional_vector_from_object(penalty0, ncon, "penalty0");
    std::vector<double> grad_theta(ntheta, 0.0);
    alm_info info;
    optimizer_solve_info backward_info;
    info.iterations = 0;
    info.final_residual = 0.0;
    info.penalty = 0.0;
    backward_info.iterations = 0;
    backward_info.final_residual = 0.0;

    int status;
    if (backward_options.enable) {
        status = alm_solve_with_backward_and_penalty0(
            &problem,
            &params,
            solution.data(),
            multipliers.data(),
            penalties0.empty() ? nullptr : penalties0.data(),
            theta_vec.data(),
            variable_vec.data(),
            &info,
            grad_theta.data(),
            &backward_info);
    } else {
        status = alm_solve_with_penalty0(
            &problem,
            &params,
            solution.data(),
            multipliers.data(),
            penalties0.empty() ? nullptr : penalties0.data(),
            theta_vec.data(),
            variable_vec.data(),
            &info);
    }

    g_compiled_oracle = nullptr;
    g_constraint_penalty_scale = 1.0;
    g_constraint_penalty_max = 0.0;

    if (status == FAILURE && info.iterations == 0) {
        throw std::runtime_error("alm_solve failed");
    }
    if (status < 0) {
        throw std::runtime_error("alm_solve backward failed");
    }

    py::dict result;
    result["solution"] = array_from_vector(solution);
    result["multipliers"] = array_from_vector(multipliers);
    result["iterations"] = info.iterations;
    result["final_residual"] = info.final_residual;
    result["penalty"] = info.penalty;
    result["forward_time_sec"] = info.forward_time_sec;
    result["backward_time_sec"] = info.backward_time_sec;
    result["inner_iterations"] = current_alm_inner_iterations();
    result["penalties"] = current_alm_penalties();
    if (backward_options.enable) {
        result["grad_theta"] = array_from_vector(grad_theta);
        result["backward_iterations"] = backward_info.iterations;
        result["backward_residual"] = backward_info.final_residual;
        result["backward_peak_workspace_bytes"] =
            py::int_(panda_backward_get_last_peak_workspace_bytes());
        result["backward_solver_used"] =
            backward_solver_name(panda_backward_get_last_solver_used());
        result["backward_fallback_used"] =
            panda_backward_get_last_fallback_used() != FALSE;
    }
    return result;
}

} // namespace

PYBIND11_MODULE(_lapanda, m)
{
    py::class_<SolverOptions>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &SolverOptions::max_iterations)
        .def_readwrite("tolerance", &SolverOptions::tolerance)
        .def_readwrite("buffer_size", &SolverOptions::buffer_size)
        .def_readwrite("max_stable_iter", &SolverOptions::max_stable_iter)
        .def_readwrite("verbose", &SolverOptions::verbose);

    py::class_<BackwardOptions>(m, "BackwardOptions")
        .def(py::init<>())
        .def_readwrite("enable", &BackwardOptions::enable)
        .def_readwrite("tolerance", &BackwardOptions::tolerance)
        .def_readwrite("max_iterations", &BackwardOptions::max_iterations)
        .def_readwrite("restart", &BackwardOptions::restart)
        .def_readwrite("linear_solver", &BackwardOptions::linear_solver)
        .def_readwrite("constraint_penalty_scale", &BackwardOptions::constraint_penalty_scale)
        .def_readwrite("constraint_penalty_max", &BackwardOptions::constraint_penalty_max);

    m.def("solve_dense_minres", &solve_dense_minres,
          py::arg("matrix"),
          py::arg("rhs"),
          py::arg("tolerance"),
          py::arg("max_iterations"));

    py::class_<AlmOptions>(m, "AlmOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &AlmOptions::max_iterations)
        .def_readwrite("tolerance", &AlmOptions::tolerance)
        .def_readwrite("initial_penalty", &AlmOptions::initial_penalty)
        .def_readwrite("penalty_update_factor", &AlmOptions::penalty_update_factor)
        .def_readwrite("max_penalty", &AlmOptions::max_penalty)
        .def_readwrite("sufficient_decrease_factor", &AlmOptions::sufficient_decrease_factor)
        .def_readwrite("verbose", &AlmOptions::verbose)
        .def_readwrite("warm_start_inner", &AlmOptions::warm_start_inner);

    py::class_<PythonOracle>(m, "Oracle")
        .def(py::init<>())
        .def_readwrite("n", &PythonOracle::n)
        .def_readwrite("ntheta", &PythonOracle::ntheta)
        .def_readwrite("nvar", &PythonOracle::nvar)
        .def_readwrite("ncon", &PythonOracle::ncon)
        .def_readwrite("cost_gradient", &PythonOracle::cost_gradient)
        .def_readwrite("prox", &PythonOracle::prox)
        .def_readwrite("jprox", &PythonOracle::jprox)
        .def_readwrite("hvp", &PythonOracle::hvp)
        .def_readwrite("loss_grad", &PythonOracle::loss_grad)
        .def_readwrite("vjp", &PythonOracle::vjp)
        .def_readwrite("constraint", &PythonOracle::constraint)
        .def_readwrite("constraint_jtprod", &PythonOracle::constraint_jtprod)
        .def_readwrite("constraint_hvp", &PythonOracle::constraint_hvp)
        .def_readwrite("constraint_vjp", &PythonOracle::constraint_vjp);

    m.def("solve_panda", &solve_panda_impl,
          py::arg("oracle"),
          py::arg("x0"),
          py::arg("theta"),
          py::arg("variable"),
          py::arg("solver_options") = SolverOptions(),
          py::arg("backward_options") = BackwardOptions());

    m.def("solve_panda_compiled", &solve_panda_compiled_impl,
          py::arg("library_path"),
          py::arg("n"),
          py::arg("ntheta"),
          py::arg("nvar"),
          py::arg("box_lower"),
          py::arg("box_upper"),
          py::arg("x0"),
          py::arg("theta"),
          py::arg("variable"),
          py::arg("solver_options") = SolverOptions(),
          py::arg("backward_options") = BackwardOptions());

    m.def("solve_lapanda", &solve_alm_impl,
          py::arg("oracle"),
          py::arg("x0"),
          py::arg("theta"),
          py::arg("variable"),
          py::arg("constraint_lower"),
          py::arg("constraint_upper"),
          py::arg("inner_solver_options") = SolverOptions(),
          py::arg("alm_options") = AlmOptions(),
          py::arg("backward_options") = BackwardOptions(),
          py::arg("multiplier0") = py::none(),
          py::arg("penalty0") = py::none());

    m.def("solve_lapanda_compiled", &solve_alm_compiled_impl,
          py::arg("library_path"),
          py::arg("n"),
          py::arg("ntheta"),
          py::arg("nvar"),
          py::arg("ncon"),
          py::arg("box_lower"),
          py::arg("box_upper"),
          py::arg("x0"),
          py::arg("theta"),
          py::arg("variable"),
          py::arg("constraint_lower"),
          py::arg("constraint_upper"),
          py::arg("inner_solver_options") = SolverOptions(),
          py::arg("alm_options") = AlmOptions(),
          py::arg("backward_options") = BackwardOptions(),
          py::arg("multiplier0") = py::none(),
          py::arg("penalty0") = py::none());

}
