#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/arithmetic/intrinsic_v.cuh>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

using cu::tangents;

#define PRINT_DEBUG 1
#if PRINT_DEBUG
#include <iostream>
#endif

constexpr __device__ auto f(auto x, auto y, auto z, auto w)
{
    auto print = [](auto &x) {
        printf("f {%g, cv [%g, %g, %g, %g] cc [%g, %g, %g, %g] lb [%g, %g, %g, %g] ub [%g, %g, %g, %g]}\n",
               x.cv.v,
               x.cv.ds[0], x.cv.ds[1], x.cv.ds[2], x.cv.ds[3],
               x.cc.ds[0], x.cc.ds[1], x.cc.ds[2], x.cc.ds[3],
               x.box.lb.ds[0], x.box.lb.ds[1], x.box.lb.ds[2], x.box.lb.ds[3],
               x.box.ub.ds[0], x.box.ub.ds[1], x.box.ub.ds[2], x.box.ub.ds[3]);
    };

    // printf("f {%g, [%g, %g], [%g, %g], [%g, %g]}\n", x.cv.v, x.cv.ds[0], x.cv.ds[1], y.cv.v, y.cv.ds[1], z.cv.v, z.cv.ds[2]);

    // auto a     = x * y + z + w;
    // auto a = x + y;
    // cu::mccormick<tangents<double, N>> a;

    print(x);
    print(y);
    auto a = x + y;
    // a.cv = x.cv + y.cv;
    // a.cc = x.cc + y.cc;
    // a.box.lb = x.box.lb + y.box.lb;
    // a.box.ub = x.box.ub + y.box.ub;

    // auto a = z + w;
    // auto a = x * y;
    print(a);
    // auto a = x;

    return a;
}

#if 0
// N is the number or simultaneous tangent computations
// M is the number of mccormick relaxations to compute per block
// *in stores the input values in SOA format
// *out stores the output values in SOA format (including the derivatives!)
// n_elems the number of elements
// n_vars the number of variables
template<typename T, int N, int M>
__global__ void kernel(T *in, T *out, int n_elems, int n_vars)
{
    extern __shared__ cu::mccormick<cu::tangents<T, N>> xs[];

    int n        = n_elems * n_vars;
    int gid      = threadIdx.x + blockIdx.x * blockDim.x; // global id
    int bid      = blockIdx.x;                            // block id
    int tid      = threadIdx.x;                           // thread id inside block
    int gridSize = blockDim.x * gridDim.x;

#if PRINT_DEBUG
    if (gid == 0) {
        printf("n is: %d\n", n);
        printf("blockdim: %d\n", blockDim.x);
        printf("griddim: %d\n", gridDim.x);
        printf("grid stride is: %d\n", blockDim.x * gridDim.x);

        // for (int i = 0; i < M; i++) {
        //     xs[i].cv.v = 69;
        //     printf("accessed xs[%d]: %g\n", i, xs[i].cv.v);
        // }
    }
#endif

    // Copy from global to shared memory
    if (tid < 4 * M) {
        for (int i = gid; i < n * 4; i += gridSize) { // * 4 because of cv, cc, lb, ub in mcormick
            int sid = tid * (N + 1);                  // shared memory id

#if PRINT_DEBUG
            printf("[gid:%3d][bid:%3d][tid:%3d] copy global [%3d] into shared [%3d] (bank: [%3d]) value: %g\n",
                   gid, bid, tid, i, sid, sid % 32, in[i]);
#endif

            ((double *)xs)[sid] = in[i]; // init value
        }
    } 

    // else if (tid < 64) { // seed cv
    //     // tid in [32, 63]
    //
    //
    //     // init derivatives in separate warp
    //     // for (int j = 1; j <= N; j++) {
    //     // ((double *)xs)[sid + j] = 0.0;
    //     // }
    //
    //     // int sid             = tid - 31;
    //     int sid = tid - 32 + 1;
    //
    //     if (sid <= N) {
    //         ((double *)xs)[sid] = 42.0;
    //         printf("[gid:%3d][bid:%3d][tid:%3d] sid %d %g\n", gid, bid, tid, sid, ((double *)xs)[sid]);
    //     }
    //
    // } else if (tid < 96) { // seed cc
    //
    //     int sid = tid - 32 + 1;
    //
    //     if (sid <= N + 32) {
    //         ((double *)xs)[sid] = 69.0;
    //         printf("[gid:%3d][bid:%3d][tid:%3d] sid %d %g\n", gid, bid, tid, sid, ((double *)xs)[sid]);
    //     }
    // } else if (tid < 128) { // seed lb and ub
    //
    //     int sid = tid - 32 + 1;
    //
    //     if (sid <= N + 64) {
    //         ((double *)xs)[sid] = 33.0;
    //         printf("[gid:%3d][bid:%3d][tid:%3d] sid %d %g\n", gid, bid, tid, sid, ((double *)xs)[sid]);
    //     }
    // }

    // int sid = tid - 31;

    __syncthreads();

    // seed all the derivatives before doing the computation

    // if (gid < n * 4)
    //     printf("[gid:%3d][bid:%3d][tid:%3d] xs.cv.v: %g\n", gid, bid, tid, ((double *)xs)[tid * (N + 1)]);

    // __syncthreads();

    // if (bid == 0) {
    //     if (gid < n)
    //         printf("block %d accessed xs[%d]: %g\n", bid, tid, xs[tid].cv.v);
    // }

    // if (gid < n) {
    //     if (tid == 0) {
    //         for (int i = 0; i < 4; i++) {
    //             printf("mccormick: %g, %g, %g, %g\n", xs[i].cv.v, xs[i].cc.v, xs[i].box.lb.v, xs[i].box.ub.v);
    //             for (int j = 0; j < N; j++) {
    //                 printf("[gid:%3d][bid:%3d][tid:%3d] cv ds[%d]: %g\n", gid, bid, tid, j, xs[i].cv.ds[j]);
    //             }
    //         }
    //     }
    // }

    // for (int i = gid; i < n; i += gridSize) { // * 4 because of cv, cc, lb, ub in mcormick
    //     auto res = f(xs[tid], xs[tid + 1], xs[tid + 2], xs[tid + 3]);
    // }

    // for (int i = gid; i < n * 4; i += gridSize) { // * 4 because of cv, cc, lb, ub in mcormick
    //     int sid = tid * (N + 1);                  // shared memory id
    //     // printf("[gid:%3d][bid:%3d][tid:%3d] copy global [%3d] into shared [%3d] (bank: [%3d]) value: %g\n",
    //     //        gid, bid, tid, i, sid, sid % 32, in[i]);
    //     //
    //     // ((double *)xs)[sid] = in[i]; // init value
    //
    //     // for (int j = 1; j <= N; j++) { // init derivatives
    //     //     ((double *)xs)[sid + j] = 0.0;
    //     // }
    // }

#if 0
    // we get a single output mccormick<tangent> for each element
    for (int i = tid; i < n_elems; i += blockDim.x * gridDim.x) {
        // printf("cv ith entry: %d\n", i + 0 * n_elems);
        // printf("cc ith entry: %d\n", i + 1 * n_elems);
        // printf("lb ith entry: %d\n", i + 2 * n_elems);
        // printf("ub ith entry: %d\n", i + 3 * n_elems);
        printf("copy out: [%d] [%d] [%d] [%d]\n", i + 0 * n_elems, i + 1 * n_elems, i + 2 * n_elems, i + 3 * n_elems); 

        out[i + 0 * n_elems] = value(xs[i].cv);     // all cv values
        out[i + 1 * n_elems] = value(xs[i].cc);     // all cc values
        out[i + 2 * n_elems] = value(xs[i].box.lb); // all lb values
        out[i + 3 * n_elems] = value(xs[i].box.ub); // all ub values

        // TODO: we can combine all the tangents of the different values together...
        // for (int j = 0; j < n_vars; j++) { // all cv derivatives (ds[0], ds[0], ... ds[0], ds[1], ds[1], ... ds[1], etc.)
        //     printf("cv ds ith entry: %d, %g\n", i + (j + 1) * n + j, xs[i].cv.ds[j]);
        //     out[i + (j + 1) * n + j] = xs[i].cv.ds[j];
        // }

        // printf("ith entry: %d\n", i + (n_vars + 1) * 1 * n);
        // for (int j = 0; j < n_vars; j++) {               // all cc derivatives
        //     out[i + (n_vars + j + 1) * n + j] = xs[i].cc.ds[j];
        // }
        // // NOTE: for lb and ub we only save the values
    }
#endif
}

#endif

/*

instead of storing variables

0 1 2 3 4 5 6 7 8 9 ...

in shared memory like this:

block 0                       1
0 1 2 3 | 4 5 6 7             8 9 10 11 ...


do:
block 0            1            2
0 1 2 3            4 5 6 7      8 9 10 11
12 13 14 15 ...

*/

template<typename T, int N, int M>
__global__ void kernel(T *in, T *out, int n_elems, int n_vars)
{
    constexpr int n_out_offset = M * 4 * (N + 1);
    // constexpr int n_bytes_shared_out = n_elems * 4 * sizeof(double) * (N + 1);

    extern __shared__ cu::mccormick<cu::tangents<T, N>> xs[];

    int n                = n_elems * n_vars;
    int gid              = threadIdx.x + blockIdx.x * blockDim.x; // global id
    int bid              = blockIdx.x;                            // block id
    int tid              = threadIdx.x;                           // thread id inside block
    int gridSize         = blockDim.x * gridDim.x;
    int n_blocks         = gridDim.x;
    int n_doubles_per_mc = 4 * (N + 1);

    // printf("n blocks is: %d\n", n_blocks);

    int vid = floor(gid / n_doubles_per_mc); // variable id
    // if (gid < n * n_doubles_per_mc)
    //     printf("[gid:%3d][bid:%3d][tid:%3d] vid: %d\n", gid, bid, tid, vid);

    int tangent_idx = gid % (N + 1) - 1; // tangent index for this thread, -1 is no tangent but a value to be skipped
    // if (gid < n * 4 * (N+1))
    //     printf("[gid:%3d][bid:%3d][tid:%3d] tangent_idx: %d\n", gid, bid, tid, tangent_idx);

    int xid = floor(gid / n_vars); // mccormick id in xs

    // seed tangents
    if (gid < n * n_doubles_per_mc) {
        bool is_cv_or_cc    = gid % n_doubles_per_mc < 2 * (N + 1);
        ((double *)xs)[tid] = (vid % n_vars == tangent_idx) && is_cv_or_cc ? 1.0 : 0.0;

        if (bid == 0) {
            printf("[gid:%3d][bid:%3d][tid:%3d][vid:%3d][tangent_idx:%3d][xid:%3d] tangent seed value: %g\n", gid, bid, tid, vid, tangent_idx, xid, ((double *)xs)[tid]);
            // printf("[gid:%3d][bid:%3d][tid:%3d] xs[0] seed value: %g\n", gid, bid, tid, xs[0].cv.ds[0]);
            // printf("[gid:%3d][bid:%3d][tid:%3d] xs[5] seed value: %g\n", gid, bid, tid, xs[5].cv.ds[1]);
        }
    }
    __syncthreads(); // TODO: might not be needed

    // We take 2 consecutive mccormick relaxation inputs to fill a complete warp.
    // When we store an 8 byte value into shared memory the result is that 16 threads of the warp will occupy
    // all 32 of the 32 bit banks (one transaction). The second 16 threads does so the same in the second transaction.
    // So, no bank conflicts should occur. See:
    // https://stackoverflow.com/questions/50787419/strategy-for-minimizing-bank-conflicts-for-64-bit-thread-separate-shared-memory/50792867#comment88583431_50787463
    int iid = floor(xid / 4) / 2;
    int block_destination = (iid % n_blocks);

    // int sh_offset = 0; // the offset for writing to shared memory

    // Copy value from global to shared memory
    for (int i = gid; i < n * 4; i += gridSize) { // * 4 because of cv, cc, lb, ub in mcormick
        int sid = tid * (N + 1);                  // shared memory id

#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d][blockd:%3d][vid:%3d][xid:%3d][iid:%3d] copy global [%3d] into shared [%3d] (bank: [%3d]) value: %g\n",
               gid, bid, tid, block_destination, vid, xid, iid, i, sid, sid % 32, in[i]);
#endif
        ((double *)xs)[sid] = in[i]; // init value
    }
    __syncthreads();

    // for n_elems is currently missing

#if 0   
    // using a single thread
    int i = gid * n_vars;
    if (i < n) {
        printf("xs[%d]: {%g, %g, %g, %g}\n", i, xs[i].cv.v, xs[i].cc.v, xs[i].box.lb.v, xs[i].box.ub.v);
        printf("xs[%d] is {%g, %g, %g, %g}\n", i, xs[i].cv.ds[0], xs[i].cc.ds[0], xs[i].box.lb.ds[0], xs[i].box.ub.ds[0]);

        printf("xs[%d]: {%g, %g, %g, %g}\n", i + 1, xs[i + 1].cv.v, xs[i + 1].cc.v, xs[i + 1].box.lb.v, xs[i + 1].box.ub.v);
        printf("xs[%d] is {%g, %g, %g, %g}\n", i + 1, xs[i + 1].cv.ds[1], xs[i + 1].cc.ds[1], xs[i + 1].box.lb.ds[1], xs[i + 1].box.ub.ds[1]);

        printf("xs[%d] is {%g, %g, %g, %g}\n", i + 2, xs[i + 2].cv.ds[2], xs[i + 2].cc.ds[2], xs[i + 2].box.lb.ds[2], xs[i + 2].box.ub.ds[2]);
        printf("xs[%d] is {%g, %g, %g, %g}\n", i + 3, xs[i + 3].cv.ds[3], xs[i + 3].cc.ds[3], xs[i + 3].box.lb.ds[3], xs[i + 3].box.ub.ds[3]);

        auto res = f(xs[i], xs[i + 1], xs[i + 2], xs[i + 3]); // NOTE: we could do multithreaded stuff in here with the shared memory
        printf("res is {%g, %g, %g, %g}\n", res.cv.v, res.cc.v, res.box.lb.v, res.box.ub.v);
        printf("res is {%g, %g, %g, %g}\n", res.cv.ds[0], res.cc.ds[0], res.box.lb.ds[0], res.box.ub.ds[0]);
        printf("res is {%g, %g, %g, %g}\n", res.cv.ds[1], res.cc.ds[1], res.box.lb.ds[1], res.box.ub.ds[1]);
        printf("res is {%g, %g, %g, %g}\n", res.cv.ds[2], res.cc.ds[2], res.box.lb.ds[2], res.box.ub.ds[2]);
        printf("res is {%g, %g, %g, %g}\n", res.cv.ds[3], res.cc.ds[3], res.box.lb.ds[3], res.box.ub.ds[3]);

        // put res into shared memory
        xs[M + 0] = res;
    }

#else

    // using multiple threads
    for (int i = gid; i < n_elems * N; i += gridSize) {

        int var_id = xid * n_vars;
        // int sid    = gid % N;
        int sid = tid % N;

        printf("[gid:%3d][bid:%3d][tid:%3d][sid:%3d][oid:%3d][varid:%3d][xid:%3d] arg idx %d %d %d %d\n",
               gid, bid, tid, sid, M + xid, var_id, xid, var_id, var_id + 1, var_id + 2, var_id + 3);

        auto res = f(xs[var_id], xs[var_id + 1], xs[var_id + 2], xs[var_id + 3]);
        // put res.ds[i] into shared memory
        // TODO: this could be done with more threads, we would need only one thread to call f, sync, then multi thread copy
        // TODO: avoid bank conflict for the different i. Right now i*4(N+1) maps to the same slot
        xs[M + xid].cv.ds[sid]     = res.cv.ds[sid];
        xs[M + xid].cc.ds[sid]     = res.cc.ds[sid];
        xs[M + xid].box.lb.ds[sid] = res.box.lb.ds[sid];
        xs[M + xid].box.ub.ds[sid] = res.box.ub.ds[sid];

        // printf("[gid:%3d][bid:%3d][tid:%3d][sid:%3d][oid:%3d][varid:%3d][xid:%3d] res is {%g, %g, %g, %g}\n", gid, bid, tid, sid, M + xid, var_id, xid, res.cv.v, res.cc.v, res.box.lb.v, res.box.ub.v);

        if (sid == 0) {
            // put res.v into shared memory
            xs[M + xid].cv.v     = res.cv.v;
            xs[M + xid].cc.v     = res.cc.v;
            xs[M + xid].box.lb.v = res.box.lb.v;
            xs[M + xid].box.ub.v = res.box.ub.v;
        }
    }

    // int i = gid;
    // if (i < N) {
    //     auto res = f(xs[bid], xs[bid + 1], xs[bid + 2], xs[bid + 3]); // NOTE: bid is probably not the general solution
    //
    //     // put res.ds[i] into shared memory
    //     // TODO: this could be done with more threads, we would need only one thread to call f, sync, then multi thread copy
    //     // TODO: avoid bank conflict for the different i. Right now i*4(N+1) maps to the same slot
    //     xs[M + 0].cv.ds[i]     = res.cv.ds[i];
    //     xs[M + 0].cc.ds[i]     = res.cc.ds[i];
    //     xs[M + 0].box.lb.ds[i] = res.box.lb.ds[i];
    //     xs[M + 0].box.ub.ds[i] = res.box.ub.ds[i];
    //
    //     if (i == 0) {
    //         // put res.v into shared memory
    //         xs[M + 0].cv.v     = res.cv.v;
    //         xs[M + 0].cc.v     = res.cc.v;
    //         xs[M + 0].box.lb.v = res.box.lb.v;
    //         xs[M + 0].box.ub.v = res.box.ub.v;
    //     }
    // }

#endif

    __syncthreads(); // TODO: probably not needed since the same thread is reading again? Make sure this is the case

    // Copy results from shared to global memory
    if (gid < n_elems * n_doubles_per_mc) {
        int sid  = n_out_offset + tid; // shared memory id
        out[gid] = ((double *)xs)[sid];
#if PRINT_DEBUG
        printf("[gid:%3d][bid:%3d][tid:%3d] copy shared [%3d] (bank: [%3d]) into global [%3d] value: %g\n",
               gid, bid, tid, sid, sid % 32, gid, out[gid]);
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
    constexpr int n_elems = 2;
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

#if PRINT_DEBUG
    //     for (int i = 0; i < n_elems; i++) {
    //         cu::mccormick<tangents<double, n_tangents>> res_v {};
    //         value(res_v.cv)     = res[i * n_copy_doubles_per_mc];
    //         value(res_v.cc)     = res[i * n_copy_doubles_per_mc + 1];
    //         value(res_v.box.lb) = res[i * n_copy_doubles_per_mc + 2];
    //         value(res_v.box.ub) = res[i * n_copy_doubles_per_mc + 3];
    //         std::cout << res_v << std::endl;
    //     }

    for (auto r : res) {
        std::cout << r << std::endl;
    }
#endif

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
