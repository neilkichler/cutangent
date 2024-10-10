#include "../common.h"

#include <cuda_runtime.h>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

namespace blackscholes
{

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
        return 0.5 * erfc(-x * M_SQRT1_2);
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
        return 0.5 * erfc(-x * 1.0 / std::numbers::sqrt2);
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

int main()
{
    constexpr int n = 1;

    using T = cu::tangent<cu::mccormick<double>>;
    blackscholes::parameters<T> xs[n]{};
    T res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        // double v = i + 1;

        value(xs[i].r)  = 0.01;
        value(xs[i].S0) = { .cv = 99.5, .cc = 100.5, .box = { .lb = 99.5, .ub = 100.5 } };
        // value(xs[i].tau)   = 0.01 * v;
        value(xs[i].tau)   = 3.0 / 12.0;
        value(xs[i].K)     = 95.0;
        value(xs[i].sigma) = 0.5;

        // update seeds to compute derivative w.r.t S0
        derivative(xs[i].S0) = 1.0;
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
    std::cout << "Black Scholes w.r.t. S0 (i.e., Delta): " << delta << std::endl;

    blackscholes::parameters<double> params;
    params = { 0.01, 99.5, 3.0 / 12.0, 95.0, 0.5 };
    std::cout << "Analytic Delta(S0= 99.5): " << blackscholes::delta(params) << std::endl;
    params = { 0.01, 100.0, 3.0 / 12.0, 95.0, 0.5 };
    std::cout << "Analytic Delta(S0=100.0): " << blackscholes::delta(params) << std::endl;
    params = { 0.01, 100.5, 3.0 / 12.0, 95.0, 0.5 };
    std::cout << "Analytic Delta(S0=100.5): " << blackscholes::delta(params) << std::endl;

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

    params = { 0.01, 99.5, 3.0 / 12.0, 95.0, 0.5 };
    std::cout << "Analytic Vega(S0= 99.5): " << blackscholes::vega(params) << std::endl;
    params = { 0.01, 100.0, 3.0 / 12.0, 95.0, 0.5 };
    std::cout << "Analytic Vega(S0=100.0): " << blackscholes::vega(params) << std::endl;
    params = { 0.01, 100.5, 3.0 / 12.0, 95.0, 0.5 };
    std::cout << "Analytic Vega(S0=100.5): " << blackscholes::vega(params) << std::endl;

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
