#include "../common.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <cstdint>
#include <iostream>

using u64 = std::uint64_t;

namespace cg = cooperative_groups;

template<typename T>
constexpr auto european_call_payoff(const T &S, const T &K)
{
    using std::max;
    return max(S - K, T {});
};

namespace blackscholes
{

template<typename T>
struct parameters
{
    T r;     // risk-free interest rate
    T S0;    // spot price
    T tau;   // time until maturity
    T K;     // strike price
    T sigma; // std. dev. of stock return (i.e., volatility)
};

template<typename T>
constexpr auto price_step(const T &S_t, const auto &Z_t, const auto &dt, const parameters<T> &params)
{
    auto [r, S0, tau, K, sigma] = params;

    using std::exp;
    using std::pow;
    using std::sqrt;

    return S_t * exp((r - 0.5 * pow(sigma, 2)) * dt + sigma * sqrt(dt) * Z_t);
}

// Call price given the Black-Scholes model
template<typename T>
constexpr auto call(const parameters<T> &params)
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
constexpr auto delta(const parameters<T> &params)
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

    auto dp = (log(forward_price / K) + 0.5 * pow(sigma, 2) * tau) / variance;
    return normcdf(dp);
}

// Derivative of call price w.r.t. sigma (i.e., volatility)
template<typename T>
constexpr auto vega(const parameters<T> &params)
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

template<typename T>
__device__ void reduce(T *sum, cg::thread_block &cta, cg::thread_block_tile<32> &tile32, T *res)
{
    const int VEC = 32;
    const int tid = cta.thread_rank();

    T beta = sum[tid];
    // printf("[tid:%3d] Payoff is: %g %g\n", tid, beta.v.cv, beta.v.cc);
    T temp;

    // reduction per warp
    for (int i = VEC / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp = sum[tid + i];
            beta += temp;
            sum[tid] = beta;
            // printf("[tid:%3d] Payoff is: %g %g\n", tid, sum[tid].v.cv, sum[tid].v.cc);
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (tid == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += VEC) {
            // printf("[tid:%3d] Payoff is: %g %g\n", tid, sum[i].v.cv, sum[i].v.cc);
            beta += sum[i];
        }
        // printf("[tid:%3d] Payoff is: %g %g\n", tid, beta.v.cv, beta.v.cc);
        *res += beta;
    }
    cg::sync(cta);
}

template<typename T>
__global__ void bs_monte_carlo(curandState *rng_states, blackscholes::parameters<T> *ps, auto *res, std::integral auto n_options)
{
    using std::exp;

    constexpr int n_paths = 128;
    constexpr int n_steps = 1000; // how many steps to take per path

    cg::thread_block cta             = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_blocks  = gridDim.x;
    const int n_threads = blockDim.x;

    __shared__ T payoffs[n_paths];

    if (tid < n_paths) {
        payoffs[tid] = 0.0;
        // printf("[gid:%3d] Accum payoff is: %g\n", gid, payoffs[tid]);
    }
    cg::sync(cta);

    assert(n_steps % 2 == 0 && "n_steps must be even right now");

    curandState state = rng_states[gid];
    for (int i = bid; i < n_options; i += n_blocks) {
        const auto [r, S0, tau, K, sigma] = ps[i];

        const auto dt = tau / n_steps;

        for (int j = tid; j < n_paths; j += n_threads) {

            auto S_t = S0;
            double2 Z_t;

            // potential pragma unroll here
            for (int k = 0; k < n_steps; k += 2) {
                Z_t = curand_normal2_double(&state);
                S_t = blackscholes::price_step(S_t, Z_t.x, dt, ps[i]);
                S_t = blackscholes::price_step(S_t, Z_t.y, dt, ps[i]);
            }

            auto payoff = european_call_payoff(S_t, K);
            payoffs[j]  = payoff;
            // printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] Payoff is: %g\n", gid, bid, tid, j, payoff.v.cv);
        }
        cg::sync(cta);

        T accum = 0.0;
        for (int j = 0; j < n_paths; j += n_threads) {
            reduce<T>(&payoffs[j], cta, tile32, &accum);
        }

        if (tid == 0) {
            T call_price = (accum / n_paths) * exp(-r * tau);

            res[i] = call_price;
        }

        if (tid == 0) {
            // printf("[gid:%3d][bid:%2d][tid:%3d] Option call price is: [%g, (%g, %g), %g]\n", gid, bid, tid, res[i].v.box.lb, res[i].v.cv, res[i].v.cc, res[i].v.box.ub);
        }
    }

    rng_states[gid] = state;
}

__global__ void rng_init(auto *rng_states)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(bid, tid, 0, &rng_states[gid]); // each block gets a different seed
    // printf("[gid:%3d] Seed is: %d\n", gid, rng_states[gid]);
}

int main()
{
    constexpr int n         = 1;
    constexpr int n_threads = 128;

#if 1
    using T = cu::tangent<cu::mccormick<double>>;
#else
    using T = double;
#endif
    blackscholes::parameters<T> xs[n] {};
    T res[n];

#if 1
    // generate dummy data
    for (int i = 0; i < n; i++) {
        value(xs[i].r)  = 0.01;
        value(xs[i].S0) = { .cv = 99.5, .cc = 100.5, .box = { .lb = 99.0, .ub = 101.0 } };
        // value(xs[i].S0)    = 99.5;
        value(xs[i].tau)   = 3.0 / 12.0;
        value(xs[i].K)     = 95.0;
        value(xs[i].sigma) = 0.5;
        // value(xs[i].sigma) = { .cv = 0.495, .cc = 0.505, .box = { .lb = 0.495, .ub = 0.505 } };

        // update seeds to compute derivative w.r.t S0
        derivative(xs[i].S0) = 1.0;
    }
#else
    for (int i = 0; i < n; i++) {
        xs[i].r     = 0.01;
        xs[i].S0    = 100.5;
        xs[i].tau   = 3.0 / 12.0;
        xs[i].K     = 95.0;
        xs[i].sigma = 0.5;
    }
#endif

    std::cout << "---- Computing Delta ----" << std::endl;
    std::cout << "S0: " << xs[0].S0 << std::endl;
    std::cout << "sigma: " << xs[0].sigma << std::endl;

    blackscholes::parameters<T> *d_xs;
    T *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    curandState *rng_states;
    CUDA_CHECK(cudaMalloc((void **)&rng_states, n * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(rng_states, 0, n * sizeof(curandState)));
    rng_init<<<n, n_threads>>>(rng_states);

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    bs_monte_carlo<<<n, n_threads>>>(rng_states, d_xs, d_res, n);
    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto delta = res[0];
    std::cout << "Black Scholes European call w.r.t. S0 (i.e., Delta): " << delta << std::endl;

    for (int i = 0; i < n; i++) {
        // update seeds to compute derivative w.r.t sigma (i.e., compute Vega)
        derivative(xs[i].S0)    = 0.0;
        derivative(xs[i].sigma) = 1.0;
    }

    std::cout << "---- Computing Vega ----" << std::endl;
    std::cout << "S0: " << xs[0].S0 << std::endl;
    std::cout << "sigma: " << xs[0].sigma << std::endl;

    rng_init<<<n, n_threads>>>(rng_states);
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    bs_monte_carlo<<<n, n_threads>>>(rng_states, d_xs, d_res, n);
    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto vega = res[0];
    std::cout << "Black Scholes European call w.r.t. sigma (i.e., Vega): " << vega << std::endl;

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
