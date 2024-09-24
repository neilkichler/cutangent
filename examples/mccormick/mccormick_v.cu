#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/arithmetic/intrinsic_v.cuh>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <cmath>
#include <iostream>

using cu::tangents;

#define PRINT_DEBUG 0

#define USE_VECTOR_LOAD_128 0
#define USE_VECTOR_LOAD_256 0

constexpr __device__ auto f(const auto &x, const auto &y, const auto &z, const auto &w)
{
    auto print = [](auto &x) {
#if 1 || PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d] f {v {%g, %g, %g, %g}, cv [%g, %g, %g, %g] cc [%g, %g, %g, %g] lb [%g, %g, %g, %g] ub [%g, %g, %g, %g]}\n",
               threadIdx.x + blockIdx.x * blockDim.x, blockIdx.x, threadIdx.x,
               x.cv.v,
               x.cc.v,
               x.box.lb.v,
               x.box.ub.v,
               x.cv.ds[0], x.cv.ds[1], x.cv.ds[2], x.cv.ds[3],
               x.cc.ds[0], x.cc.ds[1], x.cc.ds[2], x.cc.ds[3],
               x.box.lb.ds[0], x.box.lb.ds[1], x.box.lb.ds[2], x.box.lb.ds[3],
               x.box.ub.ds[0], x.box.ub.ds[1], x.box.ub.ds[2], x.box.ub.ds[3]);
#endif
    };

    // printf("f {%g, [%g, %g], [%g, %g], [%g, %g]}\n", x.cv.v, x.cv.ds[0], x.cv.ds[1], y.cv.v, y.cv.ds[1], z.cv.v, z.cv.ds[2]);

    // auto a     = x * y + z + w;
    // auto a = x + y;
    // cu::mccormick<tangents<double, N>> a;

    print(x);
    // print(y);
    auto a = x + y + z + w;
    // auto b = a + a + a + a;
    // auto c = b + b + b;
    // auto d = c + c;
    // print(a);

    return a;
}

template<typename T, int N>
__global__ void
kernel(T *in, T *out, int n_elems, int n_vars)
{
    extern __shared__ cu::mccormick<cu::tangents<T, N>> xs[];

    int n                    = n_elems * n_vars;                      // total number of mccormick variables across all elements
    int gid                  = threadIdx.x + blockIdx.x * blockDim.x; // global id
    int bid                  = blockIdx.x;                            // block id
    int tid                  = threadIdx.x;                           // thread id inside block
    int n_threads            = blockDim.x;                            // number of threads in a block
    int n_blocks             = gridDim.x;                             // number of blocks in the grid  TODO: should probably be power of two for fast % operation
    int n_doubles_per_mc     = 4 * (N + 1);                           // 4 for cv, cc, lb, ub
    int n_out_doubles_per_mc = 2 * (N + 1);                           // 2 for cv, cc
    int xid                  = gid / n_vars;                          // mccormick id in xs

    // block range when considering only mccormick values (for initial copy from global memory)
    // int n_elems_per_block = (n_elems + n_blocks - 1) >> int(log2(n_blocks));
    int n_elems_per_block = (n_elems + n_blocks - 1) / n_blocks;
    int block_start       = n_elems_per_block * bid * 4 * n_vars;
    int block_end         = min(block_start + n_elems_per_block * 4 * n_vars, n * 4);

    // block range when considering tangents as well
    int n_elems_per_block_with_tangents = n_elems_per_block * n_vars * n_doubles_per_mc;
    int t_block_start                   = n_elems_per_block_with_tangents * bid;
    int t_block_end                     = min(t_block_start + n_elems_per_block_with_tangents, n_elems * n_vars * n_doubles_per_mc);

    if (tid == 0)
        printf("[gid:%3d][bid:%3d][tid:%3d][xid:%3d] elems_per_block: %3d block_start: %3d block_end: %3d\n",
               gid, bid, tid, xid, n_elems_per_block, block_start, block_end);

    // seed tangents
    // TODO: unroll
    for (int i = tid + t_block_start; i < t_block_end; i += n_threads) {
        int v = i / n_doubles_per_mc;
#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][vid:%3d][xid:%3d][i:%3d] n_elems_tangent: %3d t_block_start: %3d t_block_end: %3d\n",
               gid, bid, tid, v, xid, i, n_elems_per_block_with_tangents, t_block_start, t_block_end);
#endif

        int tangent_idx = (i % (N + 1)) - 1;                                    // tangent index for this thread, -1 is no tangent but a value to be skipped
                                                                                // faster alternative: int tangent_idx = x - floor(1/(N+1) * x) * (N+1) - 1; // TODO: check if compiler figures this out
        bool is_cv_or_cc                  = i % n_doubles_per_mc < 2 * (N + 1); // 2 since we only seed cv and cc
        ((double *)xs)[i - t_block_start] = (v % n_vars == tangent_idx) && is_cv_or_cc ? 1.0 : 0.0;

#if PRINT_DEBUG
        // printf("[gid:%3d][bid:%3d][tid:%3d][vid:%3d][tangent_idx:%3d][xid:%3d][i:%3d] tangent seed value: %g\n",
        //        gid, bid, tid, v, tangent_idx, xid, i - t_block_start, ((double *)xs)[i - t_block_start]);
#endif
    }

    __syncthreads();

    // Load elements from global memory into shared memory trying to get a balanced allocation in all blocks
#if USE_VECTOR_LOAD_128
    for (int i = tid * 2 + block_start; i + 1 < block_end; i += n_threads) {
        int sid                       = (i - block_start) * (N + 1);
        double2 tmp                   = *(double2 *)&in[i]; // init value
        ((double *)xs)[sid]           = tmp.x;
        ((double *)xs)[sid + (N + 1)] = tmp.y;
        // printf("[gid:%3d][bid:%3d][tid:%3d][i:%3d] init value: %g\n", gid, bid, tid, i, in[i]);
    }
#elif USE_VECTOR_LOAD_256
    for (int i = tid * 4 + block_start; i + 3 < block_end; i += n_threads) {
        int sid                           = (i - block_start) * (N + 1);
        double4 tmp                       = *(double4 *)&in[i]; // init value
        ((double *)xs)[sid]               = tmp.x;
        ((double *)xs)[sid + (N + 1)]     = tmp.y;
        ((double *)xs)[sid + 2 * (N + 1)] = tmp.z;
        ((double *)xs)[sid + 3 * (N + 1)] = tmp.w;
        // printf("[gid:%3d][bid:%3d][tid:%3d][i:%3d] init value: %g\n", gid, bid, tid, i, in[i]);
    }
#else
    for (int i = tid + block_start; i < block_end; i += n_threads) {
        int sid             = (i - block_start) * (N + 1);
        ((double *)xs)[sid] = in[i]; // init value
        // printf("[gid:%3d][bid:%3d][tid:%3d][i:%3d] init value: %g\n", gid, bid, tid, i, in[i]);
    }
#endif

    __syncthreads();

    // Actual computation
    int compute_out_offset  = n_elems_per_block * n_vars;
    int compute_block_start = n_elems_per_block * bid * N;
    int compute_block_end   = min(compute_block_start + n_elems_per_block * N, n_elems * N);
    // TODO: unroll
    for (int i = tid + compute_block_start; i < compute_block_end; i += n_threads) {
        // TODO: lid is wrong
        int lid = xid * n_vars - bid * n_threads; // local variable id
        // int lid = bid * n_elems_per_block; // local variable id
        int sid = (i - compute_block_start) % N;     // shared memory tangent id (subtracting compute_blocK_start is not really needed since we are starting at a multiple of N)
        int rid = compute_out_offset + lid / n_vars; // result id in shared memory - offset exists to not overwrite inputs that might be used for different sets of seed tangents
        // int rid = compute_out_offset + lid; // result id in shared memory - offset exists to not overwrite inputs that might be used for different sets of seed tangents

        // int vid = xid - bid * N;
        // int vid = gid / n_threads + bid * n_elems_per_block;

        // int lid = bid * n_elems_per_block + (tid / N) * n_vars; // local variable id
        // int sid = (i - compute_block_start);         // shared memory tangent id (subtracting compute_blocK_start is not really needed since we are starting at a multiple of N)
        int vid = bid * n_elems_per_block + tid / N;

        auto res = f(xs[lid], xs[lid + 1], xs[lid + 2], xs[lid + 7]); // TODO: add offset to vars for testing
        // auto res = f(xs[lid + 4], xs[lid + 5], xs[lid + 6], xs[lid + 7]); // TODO: add offset to vars for testing
        // auto res = f(xs[lid + 3], xs[lid + 4], xs[lid + 10], xs[lid + 12]); // TODO: add offset to vars for testing

        xs[rid].cv.ds[sid]     = res.cv.ds[sid];
        xs[rid].cc.ds[sid]     = res.cc.ds[sid];
        xs[rid].box.lb.ds[sid] = res.box.lb.ds[sid];
        xs[rid].box.ub.ds[sid] = res.box.ub.ds[sid];

        if (sid % N == 0) {
            // put res.v into shared memory
            xs[rid].cv.v     = res.cv.v;
            xs[rid].cc.v     = res.cc.v;
            xs[rid].box.lb.v = res.box.lb.v;
            xs[rid].box.ub.v = res.box.ub.v;
        }
        printf("A [gid:%3d][bid:%3d][tid:%3d][rid:%3d][vid:%3d][xid:%3d][lid:%3d][sid:%3d] in.v: %g %g %g %g res is: %g %g %g %g %g\n",
               gid, bid, tid, rid, vid, xid, lid, sid,
               xs[lid].cv.v, xs[lid + 1].cv.v, xs[lid + 2].cv.v, xs[lid + 7].cv.v,
               res.cv.v,
               res.cv.ds[sid], res.cc.ds[sid], res.box.lb.ds[sid], res.box.ub.ds[sid]);
        // TODO: maybe we can load it directly to global memory to save on shared memory space?
    }

    __syncthreads(); // TODO: probably not needed since the same thread is reading again? Make sure this is the case

    // Copy results from shared to global memory
    int out_sh_mem_offset = compute_out_offset * n_doubles_per_mc;
    int out_block_start   = n_elems_per_block * bid * n_out_doubles_per_mc;
    int out_block_end     = min(out_block_start + n_elems_per_block * n_out_doubles_per_mc, n_elems * n_out_doubles_per_mc);
    // TODO: unroll
    for (int i = tid + out_block_start; i < out_block_end; i += n_threads) {
        int sid = out_sh_mem_offset + i - out_block_start;
        out[i]  = ((double *)xs)[sid];
#if 0 || PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][bstart:%3d][bend:%3d] copy shared [%3d] (bank: [%3d]) into global [%3d] value: %g\n",
               gid, bid, tid, out_block_start, out_block_end, sid, sid % 32, i, out[i]);
#endif
    }

    //
    // Other tangent iterations (if n_tangents < n_vars)
    //

#if 0
    int tangent_offset     = N;
    int tangent_out_offset = n_elems * n_out_doubles_per_mc;

    int k = 1;
    // for (int k = 1; k < n_vars / N; k++) {
    //     printf("k is: %d\n", k);
    // seed tangents
    // TODO: unroll
    for (int i = tid + t_block_start; i < t_block_end; i += n_threads) {
        int v = i / n_doubles_per_mc;
#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][vid:%3d][xid:%3d][i:%3d][k:%d] n_elems_tangent: %3d t_block_start: %3d t_block_end: %3d\n",
               gid, bid, tid, v, xid, i, k, n_elems_per_block_with_tangents, t_block_start, t_block_end);
#endif

        int tangent_idx = (i % (N + 1)) + tangent_offset - 1; // tangent index for this thread, -1 is no tangent but a value to be skipped

        if (tangent_idx < 0)
            continue;

        bool is_cv_or_cc                  = (i + tangent_offset) % n_doubles_per_mc < 2 * (N + 1); // 2 since we only seed cv and cc
        ((double *)xs)[i - t_block_start] = (v % n_vars == tangent_idx) && is_cv_or_cc ? 1.0 : 0.0;

#if 0 || PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][vid:%3d][tangent_idx:%3d][xid:%3d][i:%3d][k:%d] tangent seed value: %g\n",
               gid, bid, tid, v, tangent_idx, xid, i - t_block_start, k, ((double *)xs)[i - t_block_start]);
#endif
    }

    __syncthreads();

    // Actual computation
    // TODO: unroll
    for (int i = tid + compute_block_start; i < compute_block_end; i += n_threads) {
        int lid = xid * n_vars - bid * n_threads;    // local variable id
        int sid = (i - compute_block_start) % N;     // shared memory tangent id (subtracting compute_blocK_start is not really needed since we are starting at a multiple of N)
        int rid = compute_out_offset + lid / n_vars; // result id in shared memory - offset exists to not overwrite inputs that might be used for different sets of seed tangents

        // auto res = f(xs[lid], xs[lid + 1], xs[lid + 2], xs[lid + 3]); // TODO: add offset to vars for testing
        // auto res = f(xs[lid + 8], xs[lid + 9], xs[lid + 10], xs[lid + 11]);
        auto res = f(xs[lid + 10], xs[lid + 11], xs[lid + 12], xs[lid + 13]);

        printf("[gid:%3d][bid:%3d][tid:%3d][rid:%3d] res is : %g %g %g %g %g\n", gid, bid, tid, rid, res.cv.v, res.cv.ds[sid], res.cc.ds[sid], res.box.lb.ds[sid], res.box.ub.ds[sid]);

        xs[rid].cv.ds[sid]     = res.cv.ds[sid];
        xs[rid].cc.ds[sid]     = res.cc.ds[sid];
        xs[rid].box.lb.ds[sid] = res.box.lb.ds[sid];
        xs[rid].box.ub.ds[sid] = res.box.ub.ds[sid];
    }

    // update inside
    // for (int i = tid + out_block_start; i < out_block_end; i += n_threads) {
    //     int sid      = out_sh_mem_offset + i - out_block_start;
    //     int out_idx  = i + tangent_offset;
    //     out[out_idx] = ((double *)xs)[sid];
    // }

    // update outside
    for (int i = tid + out_block_start; i < out_block_end; i += n_threads) {
        int sid      = out_sh_mem_offset + i - out_block_start;
        int out_idx  = i + tangent_out_offset;
        out[out_idx] = ((double *)xs)[sid]; // TODO: skip value in copy (already have it)

#if 1 || PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][bstart:%3d][bend:%3d][k:%d] copy shared [%3d] (bank: [%3d]) into global [%3d] value: %g\n",
               gid, bid, tid, out_block_start, out_block_end, k, sid, sid % 32, out_idx, out[out_idx]);
#endif
    }

#endif
    // }
}

/* we really have two scenarios where the GPU should be used differently.

1. n_tangents < n_vars -> multiple runs over the function to get to all tangents.

   a) if n_elems large and n_vars large, then we need to compute the tangents in a loop on the same threads.
      On different blocks/SMs we can compute the other elements in parallel.
   b) if n_elems small and n_vars small, then we can compute the different tangents in other blocks/SMs.

2. n_tangents == n_vars -> only parallelization of elements


we should first try to make use of all the blocks individually doing one mccormick computation (1a).
*/

int main()
{
    constexpr int n_elems               = 40;
    constexpr int n_vars                = 16;
    constexpr int n                     = n_elems * n_vars;
    constexpr int n_copy_doubles_per_mc = 2 * (n_vars + 1); // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds
    // constexpr int n_copy_doubles_per_mc = 4 + 4 * n_vars; // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds

    constexpr int n_blocks  = 10;
    constexpr int n_threads = 256;

    constexpr int n_tangents = 16; // the number of tangents to perform per mccormick relaxation, a multiple of 32 is ideal

    constexpr int n_elems_per_block = std::ceil(double(n_elems) / n_blocks);
    constexpr int n_vars_per_block  = n_vars * n_elems_per_block; // the number of mccormick variables to access in shared memory per block

    assert(n_mccormick >= n_vars && "n_mccormick must be >= n_vars");
    static_assert(n_tangents <= n_vars, "n_tangents must be <= n_vars");

    cu::mccormick<double> xs[n_elems * n_vars] {};
    double res[n_elems * n_copy_doubles_per_mc] {};

    // generate dummy data
    for (int i = 0; i < n_elems * n_vars; i += n_vars) {
        double v = i + 2;
        for (int j = 0; j < n_vars; j++) {
            xs[i + j] = v + j;
        }
    }

#if PRINT_DEBUG
    for (auto x : xs) {
        std::cout << "x is: " << x << std::endl;
    }
#endif

    constexpr int n_bytes_shared_in  = n_vars_per_block * 4 * sizeof(double) * (n_tangents + 1);
    constexpr int n_bytes_shared_out = n_elems_per_block * 4 * sizeof(double) * (n_tangents + 1);
    constexpr int n_bytes_shared     = n_bytes_shared_in + n_bytes_shared_out;
    printf("n_bytes_shared = %d B\n", n_bytes_shared);

    double *d_xs;  // we only use a single double array for easier coalescing
    double *d_res; // same as above
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*res)));

    double *h_xs;
    CUDA_CHECK(cudaMallocHost(&h_xs, n * sizeof(*xs))); // 4 because of cv, cc, lb, ub
    memcpy(h_xs, xs, n * sizeof(*xs));

    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    kernel<double, n_tangents><<<n_blocks, n_threads, n_bytes_shared>>>(d_xs, d_res, n_elems, n_vars);
    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*d_res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

#if 1 || PRINT_DEBUG
    for (auto r : res) {
        std::cout << r << std::endl;
    }
#endif

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    CUDA_CHECK(cudaFreeHost(h_xs));

    return 0;
}
