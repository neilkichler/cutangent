#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/arithmetic/intrinsic_v.cuh>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangents;

constexpr __device__ auto f(auto x, auto y, auto z, auto w)
{

    auto print = [](auto x) { printf("f {%g, [%g, %g]}\n", x.v.cv, x.ds[0].cv, x.ds[1].cv); };
    // auto a     = x * y + z + w;
    auto a = x + y;
    // auto a = z + w;
    // auto a = x * y;
    // print(a);
    // auto a = x;

    return a;
}

// N is the number or simultaneous tangent computations
// n_elems the number of elements
// n_vars the number of variables

template<typename T, int N>
__global__ void kernel(cu::mccormick<T> *in, T *out, int n_elems, int n_vars)
{
    auto print = [](auto i, auto l, auto x) {
        printf("[%d][%d] cv{%g, [%g, %g]}\n", i, l, x.cv.v, x.cv.ds[0], x.cv.ds[1]);
        printf("[%d][%d] cc{%g, [%g, %g]}\n", i, l, x.cc.v, x.cc.ds[0], x.cc.ds[1]);
        printf("[%d][%d] lb{%g, [%g, %g]}\n", i, l, x.box.lb.v, x.box.lb.ds[0], x.box.lb.ds[1]);
        printf("[%d][%d] ub{%g, [%g, %g]}\n", i, l, x.box.ub.v, x.box.ub.ds[0], x.box.ub.ds[1]);
    };

    constexpr int n_elems_per_block = 2;              // Process 2 elements per block
    const int n_copy_doubles_per_mc = 4 + 2 * n_vars; // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds

    for (int q = blockIdx.x * n_elems_per_block; q < n_elems; q += gridDim.x * n_elems_per_block) {
        for (int i = q; (i < q + n_elems_per_block) and (i < n_elems); i++) {

            extern __shared__ cu::mccormick<cu::tangents<T, N>> xs[]; // have to set kernel with n_vars * sizeof(*xs)

            for (int l = 0; l < ceil(n_vars / N); l++) { // chunk wise vector tangent mode to eventually cover all basis vectors

                // seed
                for (int j = 0; j < n_vars; j++) {
                    auto global_idx = i * n_vars + j;
                    xs[j].cv.v      = in[global_idx].cv;
                    xs[j].cc.v      = in[global_idx].cc;
                    xs[j].box.lb.v  = in[global_idx].box.lb;
                    xs[j].box.ub.v  = in[global_idx].box.ub;

                    if (threadIdx.x == 0) {
                        printf("[%d][%d] global_idx: %d\n", i, l, global_idx);
                        printf("[%d][%d] xs[%d].cv.v = %g\n", i, l, j, xs[j].cv.v);
                        // printf("[%d][%d] xs[%d].cc.v = %g\n", i, l, j, xs[j].cc.v);
                        // printf("[%d][%d] xs[%d].box.lb.v = %g\n", i, l, j, xs[j].box.lb.v);
                        // printf("[%d][%d] xs[%d].box.ub.v = %g\n", i, l, j, xs[j].box.ub.v);
                    }

                    for (int k = 0; k < N; k++) {
                        T seed         = l * N + k == j ? 1.0 : 0.0;
                        xs[j].cv.ds[k] = seed;
                        xs[j].cc.ds[k] = seed;
                    }
                }

                if (threadIdx.x == 0) {
                    printf("[%d][%d] Input a.cv: {%g, [%g, %g]}\n", i, l, xs[0].cv.v, xs[0].cv.ds[0], xs[0].cv.ds[1]);
                    printf("[%d][%d] Input a.cc: {%g, [%g, %g]}\n", i, l, xs[0].cc.v, xs[0].cc.ds[0], xs[0].cc.ds[1]);
                    printf("[%d][%d] Input a.box.lb: {%g, [%g, %g]}\n", i, l, xs[0].box.lb.v, xs[0].box.lb.ds[0], xs[0].box.lb.ds[1]);
                    printf("[%d][%d] Input a.box.ub: {%g, [%g, %g]}\n", i, l, xs[0].box.ub.v, xs[0].box.ub.ds[0], xs[0].box.ub.ds[1]);
                    printf("[%d][%d] Input b: {%g, [%g, %g]}\n", i, l, xs[1].cv.v, xs[1].cv.ds[0], xs[1].cv.ds[1]);
                    printf("[%d][%d] Input c: {%g, [%g, %g]}\n", i, l, xs[2].cv.v, xs[2].cv.ds[0], xs[2].cv.ds[1]);
                    printf("[%d][%d] Input d: {%g, [%g, %g]}\n", i, l, xs[3].cv.v, xs[3].cv.ds[0], xs[3].cv.ds[1]);
                }

                cu::mccormick<tangents<T, N>> res_v = f(xs[0], xs[1], xs[2], xs[3]);
                print(i, l, res_v);

                // harvest
                int idx = i * n_copy_doubles_per_mc;
                int tid = threadIdx.x;

                constexpr int n_values_per_mc = 4; // cv, cc, lb, ub
                constexpr int n_derivs_per_mc = 2; // cv.ds[i], cc.ds[i]

                // it might be beneficial to first coalesce the memory in shared memory
                // before copying it back to global memory

                // first case to only copy the results from the seeded input variables to output
                // second case to ensure that we do not go over the size of the output vector
                if (tid < N and tid + l * N < n_vars) {
                    printf("out: idx: %d, tid: %d, l*N: %d, N: %d\n", idx, tid, l * N, N);
                    printf("outidx: %d\n", idx + (tid + l * N) * n_derivs_per_mc + n_values_per_mc);
                    printf("outidx: %d\n", idx + (tid + l * N) * n_derivs_per_mc + 1 + n_values_per_mc);
                    out[idx + (tid + l * N) * n_derivs_per_mc + n_values_per_mc]     = res_v.cv.ds[tid];
                    out[idx + (tid + l * N) * n_derivs_per_mc + 1 + n_values_per_mc] = res_v.cc.ds[tid];
                    printf("Update out[%d]: %g\n", idx + (tid + l * N) * n_derivs_per_mc + n_values_per_mc, out[idx + (tid + l * N) * n_derivs_per_mc + n_values_per_mc]);
                    printf("Update out[%d]: %g\n", idx + (tid + l * N) * n_derivs_per_mc + 1 + n_values_per_mc, out[idx + (tid + l * N) * n_derivs_per_mc + 1 + n_values_per_mc]);
                }

                if (tid == 0) {
                    out[idx + 0] = res_v.cv.v;
                    out[idx + 1] = res_v.cc.v;
                    out[idx + 2] = res_v.box.lb.v;
                    out[idx + 3] = res_v.box.ub.v;
                }
            }
        }
    }
}

int main()
{
    constexpr int n_elems               = 1;
    constexpr int n_vars                = 4;
    constexpr int n_copy_doubles_per_mc = 4 + 2 * n_vars; // the number of doubles to copy from device back to host per mccormick relaxation. Skips box derivatives. Take cv, cc, lb, ub, cv.ds, cc.ds

    cu::mccormick<double> xs[n_elems * n_vars];
    double res[n_elems * n_copy_doubles_per_mc];

    memset(xs, 0, n_elems * n_vars * sizeof(*xs));

    // generate dummy data
    for (int i = 0; i < n_elems * n_vars; i += n_vars) {
        double v  = i + 2;
        xs[i]     = v;
        xs[i + 1] = v + 1;
        xs[i + 2] = v + 2;
        xs[i + 3] = v + 3;
    }

    for (int i = 0; i < n_elems * n_vars; i++) {
        std::cout << "x is: " << xs[i] << std::endl;
    }

    cu::mccormick<double> *d_xs;
    double *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*d_res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*xs), cudaMemcpyHostToDevice));

    kernel<double, 2><<<2, 8, n_vars * sizeof(cu::mccormick<tangents<double, 2>>)>>>(d_xs, d_res, n_elems, n_vars);

    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * n_copy_doubles_per_mc) * sizeof(*d_res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto el : res) {
        std::cout << el << std::endl;
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
