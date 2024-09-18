#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/arithmetic/intrinsic_v.cuh>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangents;

#define PRINT_DEBUG 0

constexpr __device__ auto f(auto x, auto y, auto z, auto w)
{
    auto print = [](auto &x) {
#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d] f {%g, cv [%g, %g, %g, %g] cc [%g, %g, %g, %g] lb [%g, %g, %g, %g] ub [%g, %g, %g, %g]}\n",
               threadIdx.x + blockIdx.x * blockDim.x, blockIdx.x, threadIdx.x,
               x.cv.v,
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
    print(y);
    auto a = x + y;
    print(a);

    return a;
}

template<typename T, int N, int M>
__global__ void kernel(T *in, T *out, int n_elems, int n_vars)
{
    extern __shared__ cu::mccormick<cu::tangents<T, N>> xs[];

    int n                = n_elems * n_vars;                      // total number of mccormick variables across all elements
    int gid              = threadIdx.x + blockIdx.x * blockDim.x; // global id
    int bid              = blockIdx.x;                            // block id
    int tid              = threadIdx.x;                           // thread id inside block
    int n_threads        = blockDim.x;                            // number of threads in a block
    int n_blocks         = gridDim.x;                             // number of blocks in the grid  TODO: should probably be power of two for fast % operation
    int n_doubles_per_mc = 4 * (N + 1);                           // 4 for cv, cc, lb, ub
    int vid              = tid / n_doubles_per_mc;                // variable id
    int xid              = gid / n_vars;                          // mccormick id in xs

    // block range when considering only mccormick values (for initial copy from global memory)
    int n_elems_per_block = (n_elems + n_blocks - 1) / n_blocks;
    int block_start       = n_elems_per_block * bid * 4 * n_vars;
    int block_end         = min(block_start + n_elems_per_block * 4 * n_vars, n * 4);

    // block range when considering tangents as well
    int n_elems_per_block_with_tangents = n_elems_per_block * n_vars * n_doubles_per_mc;
    int t_block_start                   = n_elems_per_block_with_tangents * bid;
    int t_block_end                     = min(t_block_start + n_elems_per_block_with_tangents, n_elems * n_vars * n_doubles_per_mc);

    for (int i = tid + t_block_start; i < t_block_end; i += n_threads) {
        int tangent_idx = i % (N + 1) - 1; // tangent index for this thread, -1 is no tangent but a value to be skipped
                                           // faster alternative: int tangent_idx = x - floor(1/(N+1) * x) * (N+1) - 1; // TODO: check if compiler figures this out
        // seed tangents
        bool is_cv_or_cc                  = i % n_doubles_per_mc < 2 * (N + 1); // 2 since we only seed cv and cc
        ((double *)xs)[i - t_block_start] = (vid % n_vars == tangent_idx) && is_cv_or_cc ? 1.0 : 0.0;

#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][vid:%3d][tangent_idx:%3d][xid:%3d][i:%3d] tangent seed value: %g\n", gid, bid, tid, vid, tangent_idx, xid, i - t_block_start, ((double *)xs)[i - t_block_start]);
#endif
    }

    // Load elements from global memory into shared memory trying to get a balanced allocation in all blocks

    // TODO: could create a range out of block start and end to remove manual for loop
    for (int i = tid + block_start; i < block_end; i += n_threads) {
        int sid             = (i - block_start) * (N + 1);
        ((double *)xs)[sid] = in[i]; // init value
        // TODO: maybe use vector load (double2 or double4) for increased throughput? But we already use doubles
    }

    __syncthreads();

    // Actual computation
    int compute_out_offset  = n_elems_per_block * n_vars;
    int compute_block_start = n_elems_per_block * bid * N;
    int compute_block_end   = min(compute_block_start + n_elems_per_block * N, n_elems * N);
    for (int i = tid + compute_block_start; i < compute_block_end; i += n_threads) {
        int lid = xid * n_vars - bid * n_threads;    // local variable id
        int sid = (i - compute_block_start) % N;     // shared memory tangent id (subtracting compute_blocK_start is not really needed since we are starting at a multiple of N)
        int rid = compute_out_offset + lid / n_vars; // result id in shared memory - offset exists to not overwrite inputs that might be used for different sets of seed tangents

        auto res = f(xs[lid], xs[lid + 1], xs[lid + 2], xs[lid + 3]);

        xs[rid].cv.ds[sid]     = res.cv.ds[sid];
        xs[rid].cc.ds[sid]     = res.cc.ds[sid];
        xs[rid].box.lb.ds[sid] = res.box.lb.ds[sid];
        xs[rid].box.ub.ds[sid] = res.box.ub.ds[sid];

        if (sid == 0) {
            // put res.v into shared memory
            xs[rid].cv.v     = res.cv.v;
            xs[rid].cc.v     = res.cc.v;
            xs[rid].box.lb.v = res.box.lb.v;
            xs[rid].box.ub.v = res.box.ub.v;
        }
        // TODO: maybe we can load it directly to global memory to save on shared memory space?
    }

    __syncthreads(); // TODO: probably not needed since the same thread is reading again? Make sure this is the case

    // Copy results from shared to global memory
    int out_sh_mem_offset = compute_out_offset * n_doubles_per_mc;
    int out_block_start   = n_elems_per_block * bid * n_doubles_per_mc;
    int out_block_end     = min(out_block_start + n_elems_per_block * n_doubles_per_mc, n_elems * n_doubles_per_mc);
    for (int i = tid + out_block_start; i < out_block_end; i += n_threads) {
        int sid = out_sh_mem_offset + i - out_block_start;
        out[i]  = ((double *)xs)[sid];
#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][bstart:%3d][bend:%3d] copy shared [%3d] (bank: [%3d]) into global [%3d] value: %g | %g\n",
               gid, bid, tid, out_block_start, out_block_end, sid, sid % 32, i, ((double *)xs)[sid], out[i]);
#endif
    }
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
    constexpr int n_elems = 12;
    constexpr int n_vars  = 4;
    // constexpr int n_copy_doubles_per_mc = 4 + 2 * n_vars; // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds
    constexpr int n_copy_doubles_per_mc = 4 + 4 * n_vars; // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds

    constexpr int n_tangents  = 4;                // the number of tangents to perform per mccormick relaxation
    constexpr int n_mccormick = n_vars * n_elems; // the number of mccormick relaxations to access in shared memory per block

    static_assert(n_mccormick >= n_vars, "n_mccormick must be >= n_vars");
    static_assert(n_tangents <= n_vars, "n_tangents must be <= n_vars");

    constexpr int n_blocks  = 4;
    constexpr int n_threads = 256;

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

    // int n_bytes_shared = min(n_mccormick, n_elems * n_vars) * 4 * sizeof(double) * (min(n_tangents, n_vars) + 1); // TODO: what if n_vars > n_tangents?
    constexpr int n_bytes_shared_in  = n_mccormick * 4 * sizeof(double) * (n_tangents + 1); // TODO: what if n_vars > n_tangents?
    constexpr int n_bytes_shared_out = n_elems * 4 * sizeof(double) * (n_tangents + 1);
    constexpr int n_bytes_shared     = n_bytes_shared_in + n_bytes_shared_out;
    printf("n_bytes_shared = %d B\n", n_bytes_shared);

    double *d_xs;  // we only use a single double array for easier coalescing
    double *d_res; // same as above
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*res)));

    // Restructure the mccormick values to have coalesced access to shared memory in the device
    // We first store all cv, then all cc values and so on, i.e. SOA instead of AOS.

    double *h_xs;
    CUDA_CHECK(cudaMallocHost(&h_xs, n_elems * n_vars * sizeof(*xs))); // 4 because of cv, cc, lb, ub
    memcpy(h_xs, xs, n_elems * n_vars * sizeof(*xs));
    // for (int i = 0; i < n_elems * n_vars * 4; i++) {
    // xs_soa[i] = ((double *)xs)[i];
    // std::cout << xs_soa[i] << std::endl;
    // }

    // double *xs_soa;
    // CUDA_CHECK(cudaMallocHost(&xs_soa, n_elems * n_vars * sizeof(*xs))); // 4 because of cv, cc, lb, ub
    // for (int i = 0; i < n_elems * n_vars; i++) {
    //     xs_soa[i + 0 * (n_elems * n_vars)] = xs[i].cv;
    //     xs_soa[i + 1 * (n_elems * n_vars)] = xs[i].cc;
    //     xs_soa[i + 2 * (n_elems * n_vars)] = xs[i].box.lb;
    //     xs_soa[i + 3 * (n_elems * n_vars)] = xs[i].box.ub;
    // }

    // #if PRINT_DEBUG
    //     for (auto x : xs_soa) {
    //         std::cout << "x[soa] is: " << x << std::endl;
    //     }
    // #endif

    CUDA_CHECK(cudaMemcpy(d_xs, h_xs, n_elems * n_vars * sizeof(*xs), cudaMemcpyHostToDevice));
    kernel<double, n_tangents, n_mccormick><<<n_blocks, n_threads, n_bytes_shared>>>(d_xs, d_res, n_elems, n_vars);
    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*d_res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto r : res) {
        std::cout << r << std::endl;
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
