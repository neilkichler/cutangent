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

#define RELAX 1

using u64 = std::uint64_t;

namespace cg = cooperative_groups;

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
// constexpr state<T> step(state<T> &state, const auto &Z_t, const auto &dt, const parameters<T> &params)
constexpr state<T> step(state<T> &state, const auto &Z_t_x, const auto &Z_t_y, const auto &dt, const parameters<T> &params)
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
    // return { .S_t = S_t * exp((r - 0.5 * v_t) * dt + sqrt(v_t * dt) * Z_t.x),
    //          .v_t = v_t + kappa * (theta - v_t) * dt + xi * sqrt(v_t * dt) * Z_t.y };

    return { .S_t = S_t * exp((r - 0.5 * v_t) * dt + sqrt(v_t * dt) * Z_t_x),
             .v_t = v_t + kappa * (theta - v_t) * dt + xi * sqrt(v_t * dt) * Z_t_y };
}

}; // namespace heston

template<typename T>
__device__ void reduce_block(T *sum, cg::thread_block &cta, cg::thread_block_tile<32> &tile32, T *res)
{
    const int VEC = 32;
    const int tid = cta.thread_rank();

    T beta = sum[tid];
    // printf("[tid:%3d] Payoff is: %g %g\n", tid, beta.v.cv, beta.v.cc);
    // printf("[tid:%3d] Payoff is: %g\n", tid, beta);
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

#if 0
template<typename T>
__global__ void heston_monte_carlo(curandState *rng_states, heston::parameters<T> *ps, auto *res, std::integral auto n_options)
{
    using std::exp;
    using std::pow;
    using std::sqrt;

    constexpr int n_paths = 4 * 1024;
    constexpr int n_steps = 1024; // how many steps to take per path

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

    curandState rng_state = rng_states[gid];
    for (int i = bid; i < n_options; i += n_blocks) {
        const auto [r, S0, tau, K, v0, rho, kappa, theta, xi] = ps[i];

        const double rho_tmp = -0.7;

        const auto dt = tau / n_steps;

        for (int j = tid; j < n_paths; j += n_threads) {

            heston::state<T> state { .S_t = S0, .v_t = v0 };
            double2 Z_t;

            // potential pragma unroll here
            for (int k = 0; k < n_steps; k += 2) {
                Z_t = curand_normal2_double(&rng_state);

                // correlate the two random numbers
                Z_t.y = rho_tmp * Z_t.x + sqrt(1.0 - pow(rho_tmp, 2)) * Z_t.y;

                state = heston::step(state, Z_t, dt, ps[i]);
            }

            auto payoff = european_call_payoff(state.S_t, K);
            payoffs[j]  = payoff;
            // printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] Payoff is: %g\n", gid, bid, tid, j, payoff.v.cv);
            // printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] Payoff is: %g\n", gid, bid, tid, j, payoff);
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

    rng_states[gid] = rng_state;
}
#endif

constexpr int N_THREADS = 256;

template<typename T>
__global__ void heston_monte_carlo(curandState *rng_states, heston::parameters<T> *ps, auto *res, std::integral auto n_options)
{
    using std::exp;
    using std::pow;
    using std::sqrt;

    constexpr int n_paths = 32 * N_THREADS;
    constexpr int n_steps = 256; // how many steps to take per path
    // constexpr int n_steps = 1024; // how many steps to take per path

    cg::thread_block cta             = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int n_blocks  = gridDim.x;
    const int n_threads = blockDim.x;

    __shared__ T payoffs[N_THREADS];

    if (tid < n_paths) {
        payoffs[tid] = 0.0;
        // printf("[gid:%3d] Accum payoff is: %g\n", gid, payoffs[tid]);
    }
    cg::sync(cta);

    assert(n_steps % 2 == 0 && "n_steps must be even right now");

    curandState rng_state = rng_states[gid];
    for (int i = bid; i < n_options; i += n_blocks) {
        const auto [r, S0, tau, K, v0, rho, kappa, theta, xi] = ps[i];

        const double rho_tmp = -0.7;

        const auto dt = tau / n_steps;

        T accum = 0.0;

        for (int j = tid, jj = 0; j < n_paths; j += n_threads, jj += n_threads) {
            // for (int j = 0; j < n_paths / n_threads; j += 1) {

            // printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] jj is: %3d\n", gid, bid, tid, j, jj);

            heston::state<T> state { .S_t = S0, .v_t = v0 };
            double2 Z_t;

            // potential pragma unroll here
            for (int k = 0; k < n_steps; k++) {
                Z_t = curand_normal2_double(&rng_state);

                // if (tid == 0) {
                //     printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] Z_t.x, y is: %g, %g\n", gid, bid, tid, j, Z_t.x, Z_t.y);
                // }

                // correlate the two random numbers
                Z_t.y = rho_tmp * Z_t.x + sqrt(1.0 - pow(rho_tmp, 2)) * Z_t.y;

                // cu::mccormick<T> rng {.lb = -1.0, .ub = 1.0, .x = Z_t.x, .y = Z_t.y};
                // cu::mccormick<T> rng { .lb = Z_t.x, .ub = Z_t.y, .box = { .lb = -1.0, .ub = 1.0 } };
                // typename T::value_type rng_x { .cv = Z_t.x - 2.0, .cc = Z_t.x + 2, .box = { .lb = Z_t.x - 2.0, .ub = Z_t.x + 2 } };
                // typename T::value_type rng_y { .cv = Z_t.y - 2.0, .cc = Z_t.y + 2, .box = { .lb = Z_t.y - 2.0, .ub = Z_t.y + 2 } };
                //
                // typename T::value_type rng_x { .cv = - 2.0, .cc = 2.0, .box = { .lb = - 2.0, .ub = 2.0 } };
                // typename T::value_type rng_y { .cv = - 2.0, .cc = 2.0, .box = { .lb = - 2.0, .ub = 2.0 } };

                typename T::value_type rng_x { .cv = Z_t.x - 2.0e-14, .cc = Z_t.x + 2.0e-14, .box = { .lb = Z_t.x - 2.0e-14, .ub = Z_t.x + 2.0e-14 } };
                typename T::value_type rng_y { .cv = Z_t.x - 2.0e-14, .cc = Z_t.x + 2.0e-14, .box = { .lb = Z_t.x - 2.0e-14, .ub = Z_t.x + 2.0e-14 } };

                // state = heston::step(state, Z_t, dt, ps[i]);
                state = heston::step(state, rng_x, rng_y, dt, ps[i]);
            }

            auto payoff  = european_call_payoff(state.S_t, K);
            payoffs[tid] = payoff;
            // printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] Payoff is: %g\n", gid, bid, tid, j, payoff.v.cv);
            // printf("[gid:%3d][bid:%2d][tid:%3d][j:%3d] Payoff is: %g\n", gid, bid, tid, j, payoff);

            cg::sync(cta);

            reduce_block<T>(payoffs, cta, tile32, &accum);
            if (tid == 0) {
#if RELAX
                auto price_preview = ((accum / (jj + n_threads)) * exp(-r * tau));
                // printf("[gid:%3d][bid:%2d][tid:%3d] price cv is: %g %g\n",
                //        gid, bid, tid, price_preview.v.cv, price_preview.v.cc);
#else
                printf("[gid:%3d][bid:%2d][tid:%3d] Accum payoff/price is: %g %g\n", gid, bid, tid, accum, (accum / (jj + n_threads)) * exp(-r * tau));
#endif
            }
        }

        if (tid == 0) {
            T call_price = (accum / n_paths) * exp(-r * tau);

            res[i] = call_price;
        }

        if (tid == 0) {
            // printf("[gid:%3d][bid:%2d][tid:%3d] Option call price is: [%g, (%g, %g), %g]\n", gid, bid, tid, res[i].v.box.lb, res[i].v.cv, res[i].v.cc, res[i].v.box.ub);
        }
    }

    rng_states[gid] = rng_state;
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
    constexpr int n_threads = N_THREADS;

#if RELAX
    using T = cu::tangent<cu::mccormick<double>>;
#else
    using T = double;
#endif
    heston::parameters<T> xs[n] {};
    T res[n];

#if RELAX
    // generate dummy data
    for (int i = 0; i < n; i++) {
        value(xs[i].r) = 0.0319;
        // value(xs[i].S0) = { .cv = 100.0, .cc = 101.0, .box = { .lb = 100.0, .ub = 101.0 } };
        value(xs[i].S0) = { .cv = 100.0, .cc = 100.0, .box = { .lb = 99.0, .ub = 101.0 } };
        // value(xs[i].S0) = { .cv = 99.0, .cc = 100.0, .box = { .lb = 99.0, .ub = 100.0 } };
        // value(xs[i].S0) = { .cv = 99.0, .cc = 101.0, .box = { .lb = 99.0, .ub = 101.0 } };
        // value(xs[i].S0) = { .cv = 99.999999, .cc = 100.000001, .box = { .lb = 99.9, .ub = 100.001 } };
        // value(xs[i].S0) = { .cv = 99.999999, .cc = 100.000001, .box = { .lb = 99.9, .ub = 100.001 } };
        // value(xs[i].S0) = { .cv = 99.9999, .cc = 100.0001, .box = { .lb = 99.9, .ub = 100.001 } };
        // value(xs[i].S0)  = 100.0;
        value(xs[i].tau) = 1.0;
        value(xs[i].K)   = 100.0;
        value(xs[i].v0)  = 0.010201;
        // value(xs[i].v0) = 0.15;
        // value(xs[i].v0) = { .cv = 0.010201, .cc = 0.010201, .box = { .lb = 0.01, .ub = 0.011 } };

        value(xs[i].rho)   = -0.7;
        value(xs[i].kappa) = 6.21;
        // value(xs[i].kappa) = { .cv = 6.21, .cc = 6.21, .box = { .lb = 6.20999999, .ub = 6.201000001 } };
        value(xs[i].theta) = 0.019;
        // value(xs[i].theta) = { .cv = 0.019, .cc = 0.019, .box = { .lb = 0.018, .ub = 0.02 } };

        value(xs[i].xi)    = 0.61;
        //
        // value(xs[i].xi)    = 0.4;
        // value(xs[i].xi) = 0.2;
        // value(xs[i].xi) = { .cv = 0.2, .cc = 0.2, .box = { .lb = 0.19, .ub = 0.21 } };

        // value(xs[i].r)     = 0.02;
        // value(xs[i].S0)    = 55.0;
        // value(xs[i].tau)   = 1.0;
        // value(xs[i].K)     = 50.0;
        // value(xs[i].v0)    = 0.04;
        // value(xs[i].rho)   = -0.7;
        // value(xs[i].kappa) = 2.0;
        // value(xs[i].theta) = 0.04;
        // value(xs[i].xi)    = 0.3;

        // update seeds to compute derivative w.r.t S0
        derivative(xs[i].S0) = 1.0;
    }
#else
    for (int i = 0; i < n; i++) {
        // xs[i].r     = 0.02;
        // xs[i].S0    = 55.0;
        // xs[i].tau   = 1.0;
        // xs[i].K     = 50.0;
        // xs[i].v0    = 0.04;
        // xs[i].rho   = -0.7;
        // xs[i].kappa = 2.0;
        // xs[i].theta = 0.04;
        // xs[i].xi    = 0.3;

        xs[i].r     = 0.0319;
        xs[i].S0    = 100.0;
        xs[i].tau   = 1.0;
        xs[i].K     = 100.0;
        xs[i].v0    = 0.010201;
        xs[i].rho   = -0.7;
        xs[i].kappa = 6.21;
        xs[i].theta = 0.019;
        xs[i].xi    = 0.61;
    }
#endif

    std::cout << "---- Computing Delta ----" << std::endl;
    std::cout << "S0: " << xs[0].S0 << std::endl;
    std::cout << "v0: " << xs[0].v0 << std::endl;

    heston::parameters<T> *d_xs;
    T *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    curandState *rng_states;
    CUDA_CHECK(cudaMalloc((void **)&rng_states, n * sizeof(curandState)));
    CUDA_CHECK(cudaMemset(rng_states, 0, n * sizeof(curandState)));
    rng_init<<<n, n_threads>>>(rng_states);

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    heston_monte_carlo<<<n, n_threads>>>(rng_states, d_xs, d_res, n);
    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto delta = res[0];
    std::cout << "Heston European call w.r.t. S0 (i.e., Delta): " << delta << std::endl;

#if RELAX
    for (int i = 0; i < n; i++) {
        // update seeds to compute derivative w.r.t sigma (i.e., compute Vega)
        derivative(xs[i].S0) = 0.0;
        derivative(xs[i].v0) = 1.0;
    }

    std::cout << "---- Computing Vega ----" << std::endl;
    std::cout << "S0: " << xs[0].S0 << std::endl;
    std::cout << "v0: " << xs[0].v0 << std::endl;

    rng_init<<<n, n_threads>>>(rng_states);
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    heston_monte_carlo<<<n, n_threads>>>(rng_states, d_xs, d_res, n);
    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto vega = res[0];
    std::cout << "Heston European call w.r.t. sigma (i.e., Vega): " << vega << std::endl;

#endif

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
