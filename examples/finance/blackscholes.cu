#include "../common.h"

#include <cuda_runtime.h>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>
#include <vector>

namespace blackscholes
{

constexpr auto sqrt1_2 = 1.0 / std::numbers::sqrt2;

template<typename T>
struct parameters
{
    T r;     // interest rate
    T S0;    // spot price
    T tau;   // time until maturity
    T K;     // strike price
    T sigma; // std. dev. of stock return (i.e., volatility)
};

// Call price given the Black-Scholes model
template<typename T>
constexpr auto call(parameters<T> params)
{
    auto [r, S0, tau, K, sigma] = params;
    assert((S0 > 0.0) && (tau > 0.0) && (sigma > 0.0) && (K > 0.0));

    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;

    auto normcdf = [](auto x) {
        using std::erfc;
        return 0.5 * erfc(-x * sqrt1_2);
    };

    auto discount_factor = exp(-r * tau);
    auto variance        = sigma * sqrt(tau);
    auto forward_price   = S0 / discount_factor;

    auto dp         = (log(forward_price / K) + 0.5 * pow(sigma, 2) * tau) / variance;
    auto dm         = dp - variance;
    auto call_price = discount_factor * (forward_price * normcdf(dp) - K * normcdf(dm));
    return call_price;
}

// Derivative of call price w.r.t. S0 (spot price)
template<typename T>
constexpr auto delta(parameters<T> params)
{
    auto [r, S0, tau, K, sigma] = params;
    assert((S0 > 0.0) && (tau > 0.0) && (sigma > 0.0) && (K > 0.0));

    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;

    auto normcdf = [](auto x) {
        using std::erfc;
        return 0.5 * erfc(-x * sqrt1_2);
    };

    auto discount_factor = exp(-r * tau);
    auto variance        = sigma * sqrt(tau);
    auto forward_price   = S0 / discount_factor;

    auto dp = (log(forward_price / K) + 0.5 * pow(sigma, 2) * tau) / variance;
    return normcdf(dp);
}

// Derivative of call price w.r.t. sigma (i.e., volatility)
template<typename T>
constexpr auto vega(parameters<T> params)
{
    auto [r, S0, tau, K, sigma] = params;
    assert((S0 > 0.0) && (tau > 0.0) && (sigma > 0.0) && (K > 0.0));

    using std::exp;
    using std::log;
    using std::pow;
    using std::sqrt;

    auto normpdf = [](auto x) {
        return exp(-pow(x, 2) / 2.0) / sqrt(2.0 * std::numbers::pi);
    };

    auto discount_factor = exp(-r * tau);
    auto variance        = sigma * sqrt(tau);
    auto forward_price   = S0 / discount_factor;

    auto dp = (log(forward_price / K) + 0.5 * pow(sigma, 2) * tau) / variance;

    return S0 * normpdf(dp) * sqrt(tau);
}

}; // namespace blackscholes

__global__ void bs_kernel(auto *ps, auto *res, std::integral auto n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = blackscholes::call(ps[i]);
    }
}

__global__ void bs_packed_kernel(auto *ps, auto *res, std::integral auto n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        blackscholes::parameters<decltype(ps[0])> p { ps[0], ps[1], ps[2], ps[3], ps[4] };
        res[i] = blackscholes::call(p);
    }
}

// template<typename T>
// void expand(auto &&f, T &x, bounds input_bounds, bounds output_bounds)
// {
// }
// template<typename T>
// void expand(auto &&f, T &x, bounds input_bounds, bounds output_bounds)
// {
// }

// template<typename T>
// void expand(bounds b, auto &&f, T &x, ...)
// {
// }

// template<typename T>
// struct poisoned
// {
//     T value;
// };
//
// int main()
// {
//
//     auto fn = [](auto x) { return x * x; };
//
//     // poisoned<double> p { .value = 1.0 };
//     // expand(fn, p.value, { .lb = -1.0, .ub = 1.0 }, { .lb = -1.0, .ub = 1.0 });
//
//     return 0;
// }

template<typename T>
struct bounds
{
    T lb;
    T ub;
};

template<typename T>
struct expansion_info
{
    std::vector<bounds<T>> in;
    // std::vector<bounds<cu::tangent<T>>> out;
    std::vector<cu::tangent<bounds<T>>> out;
};

template<typename T>
void cuda_fn(const T *xs, T *res, int n)
{
    T *d_xs, *d_res;
    // TODO: alloc should be done outside of fn kernel
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    // CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    // bs_packed_kernel<<<n, 1>>>(d_xs, d_res, n);
    // CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));
}

// output bounds specify how large the width of the output interval maximally is allowed to be
// input bounds specify the acceptable range of the input variables
template<typename T>
expansion_info<T> expand(/* auto &&f, */ std::vector<T> &xs,
                         const std::vector<bounds<T>> &input_bounds,
                         const std::vector<bounds<T>> &output_bounds,
                         int maxiter)
{
    using ExpandT = cu::tangent<cu::mccormick<T>>;

    T expansion_rate = 0.1;

    auto n = xs.size();
    auto m = output_bounds.size();
    std::vector<ExpandT> xs_expanded(n);
    std::vector<bounds<T>> expanded_domain(n);
    // std::vector<bounds<cu::tangent<T>>> expanded_range(n);
    std::vector<cu::tangent<bounds<T>>> expanded_range(n);

    for (auto i = 0u; i < n; i++) {
        value(xs_expanded[i]) = xs[i];
    }

    for (auto i = 0u; i < n; i++) {
        std::vector<ExpandT> expanded(m);
        for (int j = 0; j < maxiter; j++) {
            value(xs_expanded[i]).box.lb -= j * expansion_rate;
            value(xs_expanded[i]).box.ub += j * expansion_rate;

            if (value(xs_expanded[i]).box.lb < input_bounds[i].lb || value(xs_expanded[i]).box.ub > input_bounds[i].ub) {
                break;
            }

            derivative(xs_expanded[i]) = 1.0;

            // expanded = f(xs_expanded[i]);
            // f(xs_expanded.data(), expanded.data(), 1);
            cuda_fn(xs_expanded.data(), expanded.data(), 1);

            for (auto k = 0u; k < m; k++) {
                if (value(expanded[k]).cv < output_bounds[k].lb || value(expanded[k]).cc > output_bounds[k].ub) {
                    break;
                }
            }

            derivative(xs_expanded[i]) = 0.0;
        }

        expanded_domain[i] = { value(xs_expanded[i]).cv, value(xs_expanded[i]).cc };

        value(expanded_range[i])      = { value(expanded[i]).cv, value(xs_expanded[i]).cc };
        derivative(expanded_range[i]) = { derivative(expanded[i]).cv, derivative(xs_expanded[i]).cc };
    }

    return { expanded_domain, expanded_range };
}

int main()
{
    blackscholes::parameters<double> p { .r = 0.01, .S0 = 100.0, .tau = 3.0 / 12.0, .K = 95.0, .sigma = 0.5 };
    std::vector<double> ps({ p.r, p.S0, p.tau, p.K, p.sigma });

    std::vector<bounds<double>> in_bounds(ps.size());
    in_bounds[0] = { p.r, p.r };
    in_bounds[1] = { .lb = 50.0, .ub = 150.0 };
    in_bounds[2] = { p.tau, p.tau };
    in_bounds[3] = { p.K, p.K };
    in_bounds[4] = { p.sigma, p.sigma };

    std::vector<bounds<double>> out_bounds(1);
    out_bounds[0] = { .lb = 0.0, .ub = 100.0 };

    auto res = expand(ps, in_bounds, out_bounds, 10);
    // auto res = expand(cuda_fn<double>, ps, in_bounds, out_bounds, 10);
}

#if 0
int main()
{
    constexpr int n = 1;

    using T = cu::tangent<cu::mccormick<double>>;
    blackscholes::parameters<T> xs[n] {};
    T res[n];

    for (double expand = 0.0; expand < 10.0; expand += 0.1) {
        double expand_m = 100.0 - expand;
        double expand_p = 100.0 + expand;

        // generate dummy data
        for (int i = 0; i < n; i++) {
            // double v = i + 1;

            value(xs[i].r) = 0.01;
            // value(xs[i].S0) = 100.0;
            value(xs[i].S0) = { { .lb = expand_m, .cv = 100.0, .cc = 100.0, .ub = expand_p } };
            // value(xs[i].S0) = { { .lb = 99.0, .cv = 100.0, .cc = 100.0, .ub = 101.0 } };
            // value(xs[i].tau)   = 0.01 * v;
            value(xs[i].tau)   = 3.0 / 12.0;
            value(xs[i].K)     = 95.0;
            value(xs[i].sigma) = 0.5;
            // value(xs[i].sigma) = { { .lb = 0.4, .cv = 0.5, .cc = 0.5, .ub = 0.6 } };

            // update seeds to compute derivative w.r.t S0
            derivative(xs[i].S0)    = 1.0;
            derivative(xs[i].sigma) = 0.0;
        }

        std::cout << "---- Computing Delta ----" << std::endl;
        std::cout << "S0: " << xs[0].S0 << std::endl;
        std::cout << "sigma: " << xs[0].sigma << std::endl;

        blackscholes::parameters<T> *d_xs;
        T *d_res;
        CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
        CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));
        CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
        bs_kernel<<<n, 1>>>(d_xs, d_res, n);
        CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

        auto delta = res[0];
        std::cout << "Black Scholes w.r.t. S0 (i.e., Delta):\n"
                  << delta << std::endl;

        blackscholes::parameters<double> params;
        params = { 0.01, expand_m, 3.0 / 12.0, 95.0, 0.5 };
        std::cout << "Analytic Delta(S0=" << expand_m << "):" << blackscholes::delta(params) << std::endl;
        params = { 0.01, 100.0, 3.0 / 12.0, 95.0, 0.5 };
        std::cout << "Analytic Delta(S0=100.0): " << blackscholes::delta(params) << std::endl;
        params = { 0.01, expand_p, 3.0 / 12.0, 95.0, 0.5 };
        std::cout << "Analytic Delta(S0=" << expand_p << "):" << blackscholes::delta(params) << std::endl;

        // update seeds to compute derivative w.r.t sigma
        for (int i = 0; i < n; i++) {
            derivative(xs[i].sigma) = 1.0;
            derivative(xs[i].S0)    = 0.0;
        }

        std::cout << "---- Computing Vega ----" << std::endl;
        std::cout << "S0: " << xs[0].S0 << std::endl;
        std::cout << "sigma: " << xs[0].sigma << std::endl;

        CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
        bs_kernel<<<n, 1>>>(d_xs, d_res, n);
        CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

        auto vega = res[0];
        std::cout << "Black Scholes w.r.t. sigma (i.e., Vega): " << vega << std::endl;

        params = { 0.01, expand_m, 3.0 / 12.0, 95.0, 0.5 };
        std::cout << "Analytic Vega(S0=" << expand_m << "):" << blackscholes::vega(params) << std::endl;
        params = { 0.01, 100.0, 3.0 / 12.0, 95.0, 0.5 };
        std::cout << "Analytic Vega(S0=100.0): " << blackscholes::vega(params) << std::endl;
        params = { 0.01, expand_p, 3.0 / 12.0, 95.0, 0.5 };
        std::cout << "Analytic Vega(S0=" << expand_p << "):" << blackscholes::vega(params) << std::endl;
        std::cout << "===========================" << std::endl;

        CUDA_CHECK(cudaFree(d_xs));
        CUDA_CHECK(cudaFree(d_res));
    }
    return 0;
}
#endif
