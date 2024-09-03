#include "../common.h"
#include "../tests/tests_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <iostream>

template<typename T>
using mc = cu::mccormick<T>;

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
    constexpr int n = 256;

    using T = mc<double>;
    blackscholes::parameters<T> xs[n];
    T res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        double v = i + 1;

        xs[i] = {
            .r     = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } },
            .S0    = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } },
            .tau   = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } },
            .K     = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } },
            .sigma = { .cv = v, .cc = v, .box = { .lb = v, .ub = v } },
        };
    }

    blackscholes::parameters<T> *d_xs;
    T *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));

    bs_kernel<<<n, 1>>>(d_xs, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto r = res[0];
    std::cout << "Black Scholes" << r << std::endl;

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
