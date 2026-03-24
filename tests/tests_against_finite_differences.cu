#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include "tests_against_finite_differences.h"
#include "tests_common.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

template<typename T>
struct domain
{
    T inf { -std::numeric_limits<T>::infinity() };
    T sup { std::numeric_limits<T>::infinity() };

    constexpr bool contains(T x) const { return x > inf && x < sup; }
};

template<typename T>
constexpr T finite_difference(auto f, T x, domain<T> dom = {})
{
    T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(std::abs(x), T(1));

    if (x - h < dom.inf) {
        return (f(x + h) - f(x)) / h; // forward difference
    } else if (x + h > dom.sup) {
        return (f(x) - f(x - h)) / h; // backward difference
    } else {
        return (f(x + h) - f(x - h)) / (2.0 * h); // central difference
    }
};

template<typename T>
__global__ void compare_unary(auto f, const cu::tangent<T> *x, T *res, T *ref, std::size_t n, domain<T> dom)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (dom.contains(value(x[i]))) {
            res[i] = derivative(f(x[i]));
            ref[i] = finite_difference([&](auto x) { return f(x); }, value(x[i]));
        }
    }
}

template<typename T>
__global__ void compare_binary(auto f, const cu::tangent<T> *a, const cu::tangent<T> *b, T *c_t, T *c_ref, std::size_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c_t[i] = derivative(f(a[i], b[i]));

        T a_v    = value(a[i]);
        T b_v    = value(b[i]);
        c_ref[i] = finite_difference([&](auto x) { return f(x, b_v); }, a_v);
    }
}

constexpr bool close_enough(double a, double b, double rtol = 5e-8)
{
    if (std::isnan(a) && std::isnan(b))
        return true;

    return std::abs(a - b) < rtol * std::max(std::max(std::abs(a), std::abs(b)), 1.0);
};

enum class op_type
{
    unary,
    binary
};

template<op_type OP_TYPE, typename T, std::size_t N>
int check_equal(const cu::tangent<T> *xs, const cu::tangent<T> *ys,
                const std::array<T, N> &res, const std::array<T, N> &ref,
                const char *name, std::ostream &out = std::cout)
{
    out << std::setw(5) << name << ':';
    const char *space = "   ";
    const char *red   = "\033[31m";
    const char *green = "\033[32m";
    const char *reset = "\033[0m";

    int errors  = 0;
    bool passed = true;
    std::stringstream ss;
    for (std::size_t i = 0; i < N; ++i) {
        if (!(close_enough(res[i], ref[i]))) {
            passed = false;
            errors++;
            ss << '\n'
               << red << "for" << reset;
            ss << space << "   x  = " << value(xs[i]) << '\n';
            if (OP_TYPE == op_type::binary) {
                ss << space << space << "   y  = " << value(ys[i]) << '\n';
            }
            ss << std::format("{0}f_tan(x) = {1}\n{0}f_cfd(x) = {2}\n", space, res[i], ref[i]);
        }
    }

    if (passed) {
        out << ' ' << green << "passed";
    } else {
        out << ' ' << red << "mismatch";
    }
    out << '\n'
        << reset << ss.str();
    return errors;
}

#define unary_fn(f)  [=] __device__(auto x) { return f(x); }
#define binary_fn(f) [=] __device__(auto x, auto y) { return f(x, y); }

#define TEST_UNARY(name, dom)                                                                      \
    compare_unary<<<1, n>>>(unary_fn(name), d_xs.data(), d_res_t.data(), d_res_fd.data(), n, dom); \
    res_t  = d_res_t;                                                                              \
    res_fd = d_res_fd;                                                                             \
    errors += check_equal<op_type::unary>(xs, ys, res_t, res_fd, #name);

#define TEST_BINARY(name)                                                                                    \
    compare_binary<<<1, n>>>(binary_fn(name), d_xs.data(), d_ys.data(), d_res_t.data(), d_res_fd.data(), n); \
    res_t  = d_res_t;                                                                                        \
    res_fd = d_res_fd;                                                                                       \
    errors += check_equal<op_type::binary>(xs, ys, res_t, res_fd, #name);

int test_against_finite_differences()
{
    using T = double;

    constexpr T pi  = std::numbers::pi;
    constexpr int n = 16;

    cu::tangent<T> xs[n] = {
        //  v,   d
        { 1.0, 1.0 },
        { 2.0, 1.0 },
        { 3.0, 1.0 },
        { 4.0, 1.0 },
        { 5.0, 1.0 },
        { 6.0, 1.0 },
        { 7.0, 1.0 },
        { 8.0, 1.0 },
        { 9.0, 1.0 },
        { 10.0, 1.0 },
        { 11.0, 1.0 },
        { 12.0, 1.0 },
        { pi, 1.0 },
        { 1.0, 1.0 },
        { 2 * pi, 1.0 },
        { 2 * pi, 1.0 },
    };

    cu::tangent<T> ys[n] = {
        //  v    d
        { 11.0, 0.0 },
        { 10.0, 0.0 },
        { 9.0, 0.0 },
        { 8.0, 0.0 },
        { 7.0, 0.0 },
        { 6.0, 0.0 },
        { 5.0, 0.0 },
        { 4.0, 0.0 },
        { 3.0, 0.0 },
        { 2.0, 0.0 },
        { 1.0, 0.0 },
        { 0.0, 0.0 },
        { 1.0, 0.0 },
        { pi, 0.0 },
        { pi, 0.0 },
        { 1.0, 0.0 },
    };

    cu::test::array<cu::tangent<T>, n> d_xs;
    cu::test::array<cu::tangent<T>, n> d_ys;
    cu::test::array<T, n> d_res_t;
    cu::test::array<T, n> d_res_fd;

    d_xs = xs;
    d_ys = ys;

    std::array<T, n> res_t;
    std::array<T, n> res_fd;

    auto pos   = [] __device__(auto x) { return +x; };
    auto neg   = [] __device__(auto x) { return -x; };
    auto recip = [] __device__<typename T>(T x) { if constexpr (std::floating_point<T>) return 1.0 / x; else return cu::recip(x); };

    int errors = 0;

    domain<T> entire;
    domain<T> positive { 0.0, std::numeric_limits<T>::infinity() };
    domain<T> one { -1.0, 1.0 };

    TEST_UNARY(pos, entire);
    TEST_UNARY(neg, entire);
    TEST_UNARY(recip, entire);
    TEST_UNARY(abs, entire);
    TEST_UNARY(sqrt, entire);
    TEST_UNARY(cbrt, entire);
    TEST_UNARY(exp, entire);
    TEST_UNARY(log, positive);
    TEST_UNARY(log2, positive);
    TEST_UNARY(log10, positive);
    TEST_UNARY(sin, entire);
    TEST_UNARY(cos, entire);
    TEST_UNARY(tan, entire);
    TEST_UNARY(asin, one);
    TEST_UNARY(acos, one);
    TEST_UNARY(atan, positive);
    TEST_UNARY(sinh, entire);
    TEST_UNARY(cosh, entire);
    TEST_UNARY(tanh, entire);
    TEST_UNARY(erf, entire);
    TEST_UNARY(erfc, entire);

    auto add = [] __device__(auto x, auto y) { return x + y; };
    auto sub = [] __device__(auto x, auto y) { return x - y; };
    auto mul = [] __device__(auto x, auto y) { return x * y; };
    auto div = [] __device__(auto x, auto y) { return x / y; };

    TEST_BINARY(add);
    TEST_BINARY(sub);
    TEST_BINARY(mul);
    TEST_BINARY(div);
    TEST_BINARY(atan2);
    TEST_BINARY(pow);

    return errors;
}
