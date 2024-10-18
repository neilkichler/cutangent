#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/arithmetic/intrinsic_v.cuh>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <cmath>
#include <iostream>

#define USE_VECTOR_LOAD_128 0
#define USE_VECTOR_LOAD_256 0

constexpr __device__ auto f(const auto &x, const auto &y, const auto &z, const auto &w)
{
    auto a = x + y + z + w;
    return a;
}

template<typename T, int N>
__global__ void
tangents_of_mccormick_from_single_elem_per_block(T *in, T *out, int n_elems, int n_vars)
{
    extern __shared__ cu::mccormick<cu::tangents<T, N>> xs[];

    int n                    = n_elems * n_vars; // total number of mccormick variables across all elements
    int bid                  = blockIdx.x;       // block id
    int tid                  = threadIdx.x;      // thread id inside block
    int n_threads            = blockDim.x;       // number of threads in a block
    int n_blocks             = gridDim.x;        // number of blocks in the grid  NOTE: should preferably be power of two for fast / operation
    int n_doubles_per_mc     = 4 * (N + 1);      // 4 for cv, cc, lb, ub
    int n_out_doubles_per_mc = 2 * (N + 1);      // 2 for cv, cc

    // block range when considering only mccormick values (for initial copy from global memory)
    // int n_elems_per_block = (n_elems + n_blocks - 1) >> int(log2(n_blocks));
    int n_elems_per_block = (n_elems + n_blocks - 1) / n_blocks;
    int block_start       = n_elems_per_block * bid * 4 * n_vars;
    int block_end         = min(block_start + n_elems_per_block * 4 * n_vars, n * 4);

    // block range when considering tangents as well
    int n_elems_per_block_with_tangents = n_elems_per_block * n_vars * n_doubles_per_mc;
    int t_block_start                   = n_elems_per_block_with_tangents * bid;
    int t_block_end                     = min(t_block_start + n_elems_per_block_with_tangents, n_elems * n_vars * n_doubles_per_mc);

    // seed tangents
    for (int i = tid + t_block_start; i < t_block_end; i += n_threads) {
        int v           = i / n_doubles_per_mc;
        int tangent_idx = (i % (N + 1)) - 1;                                    // tangent index for this thread, -1 is no tangent but a value to be skipped
                                                                                // faster alternative: int tangent_idx = x - floor(1/(N+1) * x) * (N+1) - 1; // TODO: check if compiler figures this out
        bool is_cv_or_cc                  = i % n_doubles_per_mc < 2 * (N + 1); // 2 since we only seed cv and cc
        ((double *)xs)[i - t_block_start] = (v % n_vars == tangent_idx) && is_cv_or_cc ? 1.0 : 0.0;
    }

    __syncthreads();

    // Load elements from global memory into shared memory trying to get a balanced allocation in all blocks
#if USE_VECTOR_LOAD_128
    for (int i = tid * 2 + block_start; i + 1 < block_end; i += n_threads) {
        int sid                       = (i - block_start) * (N + 1);
        double2 tmp                   = *(double2 *)&in[i]; // init value
        ((double *)xs)[sid]           = tmp.x;
        ((double *)xs)[sid + (N + 1)] = tmp.y;
    }
#elif USE_VECTOR_LOAD_256
    for (int i = tid * 4 + block_start; i + 3 < block_end; i += n_threads) {
        int sid                           = (i - block_start) * (N + 1);
        double4 tmp                       = *(double4 *)&in[i]; // init value
        ((double *)xs)[sid]               = tmp.x;
        ((double *)xs)[sid + (N + 1)]     = tmp.y;
        ((double *)xs)[sid + 2 * (N + 1)] = tmp.z;
        ((double *)xs)[sid + 3 * (N + 1)] = tmp.w;
    }
#else
    for (int i = tid + block_start; i < block_end; i += n_threads) {
        int sid             = (i - block_start) * (N + 1);
        ((double *)xs)[sid] = in[i]; // init value
    }
#endif

    CUTANGENT_CONSERVATIVE_WARP_SYNC();

    int compute_out_offset  = n_elems_per_block * n_vars;

    // result id in shared memory - offset exists to not overwrite inputs that might be used for different sets of seed tangents
    int rid = compute_out_offset;

    // Actual computation
    auto res = f(xs[0], xs[1], xs[8], xs[11]);

    for (int i = tid; i < N; i += n_threads) {
        xs[rid].cv.ds[i]     = res.cv.ds[i];
        xs[rid].cc.ds[i]     = res.cc.ds[i];
        xs[rid].box.lb.ds[i] = res.box.lb.ds[i];
        xs[rid].box.ub.ds[i] = res.box.ub.ds[i];
    }

    if (tid == 0) {
        // put res.v into shared memory only once
        xs[rid].cv.v     = res.cv.v;
        xs[rid].cc.v     = res.cc.v;
        xs[rid].box.lb.v = res.box.lb.v;
        xs[rid].box.ub.v = res.box.ub.v;
    }

    CUTANGENT_CONSERVATIVE_WARP_SYNC();

    // Copy results from shared to global memory
    int out_sh_mem_offset = compute_out_offset * n_doubles_per_mc;
    int out_block_start   = n_elems_per_block * bid * n_out_doubles_per_mc;
    int out_block_end     = min(out_block_start + n_elems_per_block * n_out_doubles_per_mc, n_elems * n_out_doubles_per_mc);

    for (int i = tid + out_block_start; i < out_block_end; i += n_threads) {
        int sid = out_sh_mem_offset + i - out_block_start;
        out[i]  = ((double *)xs)[sid];
    }
}

int main()
{
    constexpr int n_elems               = 8;
    constexpr int n_vars                = 32;
    constexpr int n                     = n_elems * n_vars;
    constexpr int n_copy_doubles_per_mc = 2 * (n_vars + 1); // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds

    constexpr int n_blocks  = 10;
    constexpr int n_threads = 32;

    // The number of tangents to perform per mccormick relaxation. 
    // A multiple of 32 is ideal and n_tangents=n_vars is best if it fits into shared memory
    constexpr int n_tangents = 32;

    constexpr int n_elems_per_block = std::ceil(double(n_elems) / n_blocks);
    constexpr int n_vars_per_block  = n_vars * n_elems_per_block; // the number of mccormick variables to access in shared memory per block

    static_assert(n_tangents <= n_vars, "n_tangents must be <= n_vars");
    static_assert(n_blocks >= n_elems, "n_blocks must be >= n_elems for now");
    static_assert(n_threads <= n_tangents, "it currently doesn't make sense to have more threads than tangents since we don't perform multiple elements in a block at a time");

    cu::mccormick<double> xs[n_elems * n_vars] {};
    double res[n_elems * n_copy_doubles_per_mc] {};

    // generate dummy data
    for (int i = 0; i < n_elems * n_vars; i += n_vars) {
        double v = i + 2;
        for (int j = 0; j < n_vars; j++) {
            xs[i + j] = v + j;
        }
    }

    constexpr int n_bytes_shared_in  = n_vars_per_block * 4 * sizeof(double) * (n_tangents + 1);
    constexpr int n_bytes_shared_out = n_elems_per_block * 4 * sizeof(double) * (n_tangents + 1);
    constexpr int n_bytes_shared     = n_bytes_shared_in + n_bytes_shared_out;
    printf("n_bytes_shared = %g KiB\n", n_bytes_shared / 1024.0);

    double *d_xs;  // we only use a single double array for easier coalescing
    double *d_res; // same as above
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*res)));

    double *h_xs;
    CUDA_CHECK(cudaMallocHost(&h_xs, n * sizeof(*xs)));
    memcpy(h_xs, xs, n * sizeof(*xs));

    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    tangents_of_mccormick_from_single_elem_per_block<double, n_tangents><<<n_blocks, n_threads, n_bytes_shared>>>(d_xs, d_res, n_elems, n_vars);
    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*d_res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto r : res) {
        std::cout << r << std::endl;
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    CUDA_CHECK(cudaFreeHost(h_xs));

    return 0;
}
