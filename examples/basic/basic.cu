#include <cuda.h>
#include <cuda_runtime.h>

#include <cumccormick/cumccormick.cuh>

#include <cutangent/cutangent.cuh>

template<typename T>
using mc = cu::mccormick<T>;

int main()
{
    // constexpr int n = 256;
    // using T = mc<double>;
    // T xs[n], ys[n], res[n];
    //
    // // generate dummy data
    // for (int i = 0; i < n; i++) {
    //     double v = i;
    //     xs[i] = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
    //     ys[i] = { .cv = -v, .cc = v, .box = { .lb = -v, .ub = v } };
    // }
    //
    // mc<double> *d_xs, *d_ys, *d_res;
    // CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    // CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*ys)));
    // CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));
    //
    // CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*ys), cudaMemcpyHostToDevice));
    //
    // kernel<<<n, 1>>>(d_xs, d_ys, d_res, n);
    //
    // CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));
    //
    // auto r = res[0];
    // printf("beale(0, 0) = " MCCORMICK_FORMAT "\n", r.box.lb, r.cv, r.cc, r.box.ub);
    //
    // CUDA_CHECK(cudaFree(d_xs));
    // CUDA_CHECK(cudaFree(d_ys));
    // CUDA_CHECK(cudaFree(d_res));

    return 0;
}
