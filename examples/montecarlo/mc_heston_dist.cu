#include "../common.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <array>
#include <cstdint>
#include <iostream>

using u64 = std::uint64_t;

namespace cg = cooperative_groups;

thread_local double dummy = 0.0;

constexpr auto &value(auto &x) { return x; }
constexpr auto &derivative(auto &x) { return dummy; }

template<typename T>
constexpr auto european_call_payoff(const T &S, const T &K)
{
    using std::max;
    return max(S - K, T {});
};

namespace heston
{

template<typename T>
struct parameters
{
    T r;     // risk-free interest rate
    T S0;    // spot price
    T tau;   // time until maturity
    T K;     // strike price
    T v0;    // initial volatility
    T rho;   // correlation of asset and volatility
    T kappa; // mean-reversion rate
    T theta; // long run average volatility
    T xi;    // volatility of volatility
};

template<typename T>
struct state
{
    T S_t; // current asset price
    T v_t; // current volatility
};

template<typename T>
constexpr state<T> step(state<T> &state, const auto &Z_t, const auto &dt, const parameters<T> &params)
{
    auto [r, S0, tau, K, v0, rho, kappa, theta, xi] = params;
    auto [S_t, v_t]                                 = state;

    using std::abs;
    using std::exp;
    using std::max;
    using std::pow;
    using std::sqrt;

    constexpr T zero {};

    v_t = max(v_t, zero); // full truncation

    // using the Euler-Maryuama discretization scheme
    return { .S_t = S_t * exp((r - 0.5 * v_t) * dt + sqrt(v_t * dt) * Z_t.y),
             .v_t = v_t + kappa * (theta - v_t) * dt + xi * sqrt(v_t * dt) * Z_t.x };
}

}; // namespace heston

template<typename T>
__device__ void reduce_block(T *sum, cg::thread_block &cta, cg::thread_block_tile<32> &tile32, T *res)
{
    const int VEC = 32;
    const int tid = cta.thread_rank();

    T beta = sum[tid];
    T temp;

    // reduction per warp
    for (int i = VEC / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp = sum[tid + i];
            beta += temp;
            sum[tid] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (tid == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += VEC) {
            beta += sum[i];
        }
        *res += beta;
    }
    cg::sync(cta);
}

constexpr int N_THREADS = 512;

namespace monte_carlo
{
struct parameters
{
    u64 n_options; // how many option scenarios to calculate
    u64 n_paths;   // how many paths to take per monte carlo simulation
    u64 n_steps;   // how many steps to take per path
};
} // namespace monte_carlo

template<typename T>
__global__ void heston_monte_carlo(monte_carlo::parameters mc_params,
                                   curandState *rng_states,
                                   heston::parameters<T> *ps,
                                   T *tmp,
                                   T *res)
{
    using std::exp;
    using std::pow;
    using std::sqrt;

    const auto [n_options, n_paths, n_steps] = mc_params;

    cg::thread_block cta             = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_blocks  = gridDim.x;
    const int n_threads = blockDim.x;

    const u64 n_paths_per_block = n_paths / n_blocks;

    __shared__ T payoffs[N_THREADS];

    if (gid == 0) {
        printf("n_paths_per_block: %llu\n", n_paths_per_block);
        printf("n_blocks: %d \n", n_blocks);
        printf("n_paths: %llu \n", n_paths);
    }

    if (tid < n_threads) {
        payoffs[tid] = 0.0;
    }
    cg::sync(cta);

    assert(n_steps % 2 == 0 && "n_steps must be even right now");

    curandState rng_state = rng_states[gid];

    int i = 0;

    const auto [r, S0, tau, K, v0, rho_, kappa, theta, xi] = ps[i];

    const auto rho = value(rho_);

    const auto dt = tau / n_steps;

    T accum = 0.0;

    for (int j = tid; j < n_paths_per_block; j += n_threads) {
        heston::state<T> state { .S_t = S0, .v_t = v0 };
        double2 Z_t;

        for (int k = 0; k < n_steps; k++) {
            Z_t = curand_normal2_double(&rng_state);

            // correlate the two random numbers
            Z_t.y = rho * Z_t.x + sqrt(1.0 - pow(rho, 2)) * Z_t.y;

            state = heston::step(state, Z_t, dt, ps[i]);
        }

        auto payoff  = european_call_payoff(state.S_t, K);
        payoffs[tid] = payoff;

        cg::sync(cta);

        reduce_block<T>(payoffs, cta, tile32, &accum);
    }

    if (tid == 0) {
        // printf("[gid:%3d][bid:%2d][tid:%3d] Accum payoff/price is: %g \n", gid, bid, tid, accum);
        tmp[bid] = accum;
    }

    rng_states[gid] = rng_state;
}

template<typename T>
__global__ void heston_price_from_payoffs(monte_carlo::parameters mc_params,
                                          curandState *rng_states,
                                          heston::parameters<T> *ps,
                                          T *tmp,
                                          T *res)
{
    const auto [n_options, n_paths, n_steps] = mc_params;

    cg::thread_block cta             = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_blocks  = gridDim.x;
    const int n_threads = blockDim.x;

    int i = 0;

    const auto [r, S0, tau, K, v0, rho_, kappa, theta, xi] = ps[i];

    // if (tid < n_threads) {
    //     printf("[gid:%3d][bid:%2d][tid:%3d] payoff partial sum is: %g \n", gid, bid, tid, tmp[tid]);
    // }

    if (bid == 0) {
        T final_payoff_sum = 0.0;
        reduce_block<T>(tmp, cta, tile32, &final_payoff_sum);

        if (tid == 0) {
            T call_price = (final_payoff_sum / n_paths) * exp(-r * tau);

            res[0] = call_price;

            // printf("[gid:%3d][bid:%2d][tid:%3d] Final payoff sum is: %g \n", gid, bid, tid, final_payoff_sum);
            // printf("[gid:%3d][bid:%2d][tid:%3d] Final price is: %g \n", gid, bid, tid, call_price);
            // printf("[gid:%3d][bid:%2d][tid:%3d] Final price is: %g %g\n", gid, bid, tid, call_price.v, call_price.d);
            // printf("[gid:%3d][bid:%2d][tid:%3d] Final price is: %g %g\n", gid, bid, tid, res[0].v, res[0].d);
        }
    }
}

__global__ void rng_init(auto *rng_states)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(bid, tid, 0, &rng_states[gid]); // each block gets a different seed
}

int main()
{
    constexpr int n = 1;
    // constexpr int n         = 8;
    constexpr int n_threads = N_THREADS;
    constexpr int n_blocks  = 1024;

    // using T = cu::tangent<cu::mccormick<double>>;
    // #if PURE_DOUBLE
    // using T = double;
    // #elif !PURE_DOUBLE
    using T = cu::tangent<double>;
    // #endif
    heston::parameters<T> xs[n] {};
    T res[n];

    // generate dummy scenarios
    for (int i = 0; i < n; i++) {
        value(xs[i].r)     = 0.0319;
        value(xs[i].S0)    = 100.0;
        value(xs[i].tau)   = 1.0;
        value(xs[i].K)     = 100.0;
        value(xs[i].v0)    = 0.010201;
        value(xs[i].rho)   = -0.7;
        value(xs[i].kappa) = 6.21;
        value(xs[i].theta) = 0.019;
        value(xs[i].xi)    = 0.61;

        // update seeds to compute derivative w.r.t S0
        derivative(xs[i].S0) = 1.0;
    }

    // derivative(xs[0].S0) = 1.0;
    // derivative(xs[1].tau) = 1.0;
    // derivative(xs[2].K) = 1.0;
    // derivative(xs[3].v0) = 1.0;
    // derivative(xs[4].kappa) = 1.0;
    // derivative(xs[5].theta) = 1.0;
    // derivative(xs[6].xi) = 1.0;
    // derivative(xs[7].r) = 1.0;

    monte_carlo::parameters mc_params { .n_options = n,
                                        .n_paths   = 1024 * 1024,
                                        .n_steps   = 1024 };

    std::cout << "---- Computing Delta ----" << std::endl;
    std::cout << "S0: " << xs[0].S0 << std::endl;
    std::cout << "v0: " << xs[0].v0 << std::endl;

    heston::parameters<T> *d_xs;
    T *d_res;
    T *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));
    CUDA_CHECK(cudaMalloc(&d_tmp, n_blocks * sizeof(*d_tmp)));

    curandState *rng_states;
    CUDA_CHECK(cudaMalloc((void **)&rng_states, n_blocks * n_threads * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(rng_states, 0, n_blocks * n_threads * sizeof(curandState)));
    rng_init<<<n_blocks, n_threads>>>(rng_states);

#if 1
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    heston_monte_carlo<<<n_blocks, n_threads>>>(mc_params, rng_states, d_xs, d_tmp, d_res);
    heston_price_from_payoffs<<<1, n_blocks>>>(mc_params, rng_states, d_xs, d_tmp, d_res);
    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    for (auto delta : res) {
        std::cout << "Heston European call w.r.t. S0 (i.e., Delta): " << delta << std::endl;
    }

#elif 0

    constexpr int n_streams = 8;
    std::array<cudaStream_t, n_streams> streams {};
    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (int i = 0; i < 8; i++) {
        derivative(xs[0].S0)    = i == 0;
        derivative(xs[0].tau)   = i == 1;
        derivative(xs[0].K)     = i == 2;
        derivative(xs[0].v0)    = i == 3;
        derivative(xs[0].kappa) = i == 4;
        derivative(xs[0].theta) = i == 5;
        derivative(xs[0].xi)    = i == 6;
        derivative(xs[0].r)     = i == 7;

        CUDA_CHECK(cudaMemcpyAsync(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice, streams[i]));
        heston_monte_carlo<<<n_blocks, n_threads, 0, streams[i]>>>(mc_params, rng_states, d_xs, d_tmp, d_res);
        heston_price_from_payoffs<<<1, n_blocks, 0, streams[i]>>>(mc_params, rng_states, d_xs, d_tmp, d_res);
        CUDA_CHECK(cudaMemcpyAsync(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost, streams[i]));

        for (auto r : res) {
            // std::cout << "Heston European call w.r.t. S0 (i.e., Delta): " << delta << std::endl;
            std::cout << "Heston European call w.r.t. parameter " << i << ": " << r << std::endl;
        }
    }

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));
#else

    for (int i = 0; i < 8; i++) {
        derivative(xs[0].S0)    = i == 0;
        derivative(xs[0].tau)   = i == 1;
        derivative(xs[0].K)     = i == 2;
        derivative(xs[0].v0)    = i == 3;
        derivative(xs[0].kappa) = i == 4;
        derivative(xs[0].theta) = i == 5;
        derivative(xs[0].xi)    = i == 6;
        derivative(xs[0].r)     = i == 7;

        CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
        heston_monte_carlo<<<n_blocks, n_threads>>>(mc_params, rng_states, d_xs, d_tmp, d_res);
        heston_price_from_payoffs<<<1, n_blocks>>>(mc_params, rng_states, d_xs, d_tmp, d_res);
        CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

        for (auto r : res) {
            // std::cout << "Heston European call w.r.t. S0 (i.e., Delta): " << delta << std::endl;
            std::cout << "Heston European call w.r.t. parameter " << i << ": " << r << std::endl;
        }
    }


#endif

    //
    // for (int i = 0; i < n; i++) {
    //     // update seeds to compute derivative w.r.t v0 (i.e., compute Vega)
    //     derivative(xs[i].S0) = 0.0;
    //     derivative(xs[i].v0) = 1.0;
    // }
    //
    // std::cout << "---- Computing Vega ----" << std::endl;
    // std::cout << "S0: " << xs[0].S0 << std::endl;
    // std::cout << "v0: " << xs[0].v0 << std::endl;
    //
    // rng_init<<<n, n_threads>>>(rng_states);
    // CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    // heston_monte_carlo<<<n, n_threads>>>(mc_params, rng_states, d_xs, d_res);
    // CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));
    //
    // for (auto vega : res) {
    //     std::cout << "Heston European call w.r.t. sigma (i.e., Vega): " << vega << std::endl;
    // }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
