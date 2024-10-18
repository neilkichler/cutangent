#ifndef CUTANGENT_ARITHMETIC_INTRINSIC_V_CUH
#define CUTANGENT_ARITHMETIC_INTRINSIC_V_CUH

#include <cutangent/tangents.h>

#pragma nv_diagnostic push
// ignore "821: function was referenced but not defined" for e.g. add_down since
// this has to be defined by the type and is only included in an upstream project.
#pragma nv_diag_suppress 821

#undef CUTANGENT_SHARED

#if CUTANGENT_USE_SHARED_MEMORY
#define CUTANGENT_SHARED __shared__
#else
#define CUTANGENT_SHARED
#endif

namespace cu::intrinsic
{
// clang-format off
template<typename T> inline __device__ T fma_down  (T x, T y, T z);
template<typename T> inline __device__ T fma_up    (T x, T y, T z);
template<typename T> inline __device__ T add_down(const T &x, const T &y);
template<typename T> inline __device__ T add_up(const T &x, const T &y);
template<typename T> inline __device__ T sub_down  (T x, T y);
template<typename T> inline __device__ T sub_up    (T x, T y);
template<typename T> inline __device__ T mul_down  (T x, T y);
template<typename T> inline __device__ T mul_up    (T x, T y);
//     template<typename T> inline __device__ T div_down  (T x, T y);
//     template<typename T> inline __device__ T div_up    (T x, T y);
//     template<typename T> inline __device__ T median    (T x, T y);
template<typename T> inline __device__ T min       (T x, T y);
template<typename T> inline __device__ T max       (T x, T y);
//     template<typename T> inline __device__ T copy_sign (T x, T y);
//     template<typename T, typename U> inline __device__ T next_after(T x, U y);
//     template<typename T> inline __device__ T rcp_down  (T x);
//     template<typename T> inline __device__ T rcp_up    (T x);
//     template<typename T> inline __device__ T sqrt_down (T x);
//     template<typename T> inline __device__ T sqrt_up   (T x);
//     template<typename T> inline __device__ T int_down  (T x);
//     template<typename T> inline __device__ T int_up    (T x);
//     template<typename T> inline __device__ T trunc     (T x);
//     template<typename T> inline __device__ T round_away(T x);
//     template<typename T> inline __device__ T round_even(T x);
//     template<typename T> inline __device__ T exp       (T x);
//     template<typename T> inline __device__ T exp10     (T x);
//     template<typename T> inline __device__ T exp2      (T x);
//     template<typename T> inline __device__ __host__ T nan();
template<typename T> inline __device__ T neg_inf();
template<typename T> inline __device__ T pos_inf();
//     template<typename T> inline __device__ T next_floating(T x);
//     template<typename T> inline __device__ T prev_floating(T x);
//
using cu::tangents;

template<int N>
inline __device__ tangents<double,N> fma_down(tangents<double, N> x, tangents<double, N> y, tangents<double, N> z)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = fma_down(x.v, y.v, z.v);
    int i = threadIdx.x;
    res.ds[i] = fma_down(x.v, y.ds[i], fma_down(x.ds[i], y.v, z.ds[i]));
    return res;
}

template<int N>
inline __device__ tangents<double, N> fma_up(tangents<double, N> x, tangents<double, N> y, tangents<double, N> z)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = fma_up(x.v, y.v, z.v);
    int i = threadIdx.x;
    res.ds[i] = fma_up(x.v, y.ds[i], fma_up(x.ds[i], y.v, z.ds[i]));
    return res;
}

template<int N>
inline __device__ tangents<double, N> add_down(const tangents<double, N> &a, const tangents<double, N> &b)
{
    CUTANGENT_SHARED tangents<double, N> res;

    if (threadIdx.x == 0) {
        res.v = add_down(a.v, b.v);
    }

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = add_down(a.ds[i], b.ds[i]);
    }

    // if (threadIdx.x == 0) {
    //     res.v = add_down(a.v, b.v);
    // } else {
    //     for (int i = threadIdx.x - 1; i < N; i += blockDim.x) {
    //         res.ds[i] = add_down(a.ds[i], b.ds[i]);
    //     }
    // }

    // int tid = threadIdx.x;
    //
    // if (tid == 0) {
    //     res.v = add_down(a.v, b.v);
    // } else if (tid < N) {
    //     int i = tid;
    //     res.ds[i] = add_down(a.ds[i], b.ds[i]);
    // }
    //
    // __syncwarp();
    // for (int i = tid + blockDim.x; i < N; i += blockDim.x) {
    //     res.ds[i] = add_down(a.ds[i], b.ds[i]);
    // }


    CUTANGENT_CONSERVATIVE_WARP_SYNC;

    // printf("[tid:%3d] add_down\n", i);

    return res;
}

template<int N>
inline __device__ tangents<double, N> add_up(const tangents<double, N> &a, const tangents<double, N> &b)
{
    CUTANGENT_SHARED tangents<double, N> res;


    if (threadIdx.x == 0) {
        res.v = add_up(a.v, b.v);
    }

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = add_up(a.ds[i], b.ds[i]);
    }


    // // res.v = add_up(a.v, b.v);
    // if (threadIdx.x == 0) {
    //     res.v = add_up(a.v, b.v);
    // } else {
    // // int i = threadIdx.x;
    // // __syncwarp();
    //     for (int i = threadIdx.x - 1; i < N; i += blockDim.x) {
    //         res.ds[i] = add_up(a.ds[i], b.ds[i]);
    //     }
    // }


    // int tid = threadIdx.x;
    //
    // if (tid == 0) {
    //     res.v = add_up(a.v, b.v);
    // } else if (tid < N) {
    //     int i = tid;
    //     res.ds[i] = add_up(a.ds[i], b.ds[i]);
    // }
    //
    // __syncwarp();
    //
    // for (int i = tid + blockDim.x; i < N; i += blockDim.x) {
    //     res.ds[i] = add_up(a.ds[i], b.ds[i]);
    // }




    CUTANGENT_CONSERVATIVE_WARP_SYNC;
    return res;
}

template<int N>
inline __device__ tangents<double, N> sub_down(tangents<double, N> a, tangents<double, N> b)
{ 
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = sub_down(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = sub_down(a.ds[i], b.ds[i]);
    return res;
}

template<int N>
inline __device__ tangents<double, N> sub_up(tangents<double, N> a, tangents<double, N> b)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = sub_up(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = sub_up(a.ds[i], b.ds[i]);
    return res;
}

template<int N>
inline __device__ tangents<double, N> mul_down(tangents<double, N> a, tangents<double, N> b)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = mul_down(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = add_down(mul_down(a.v, b.ds[i]), mul_down(a.ds[i], b.v));

    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int bid = blockIdx.x;
    // int tid = threadIdx.x;

    // if (blockIdx.x == 1) {
    //     printf("[gid:%3d][bid:%3d][tid:%3d] a.v = %g a.ds[%d] = %g a.ds[%d] = %g, b.v = %g b.ds[%d] = %g b.ds[%d] = %g, res.v = %g ds[%d] = %g ds[%d] = %g\n", 
    //            gid, bid, tid, a.v,
    //            i, a.ds[i],
    //            i, a.ds[i],
    //            b.v,
    //            i, b.ds[i],
    //            i, b.ds[i],
    //            res.v,
    //            i, res.ds[i],
    //            i, res.ds[i]);
    // }

    return res;
}

template<int N>
inline __device__ tangents<double, N> mul_up(tangents<double, N> a, tangents<double, N> b)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = mul_up(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = add_up(mul_up(a.v, b.ds[i]), mul_up(a.ds[i], b.v));
    return res;
}

template<int N>
inline __device__ tangents<double, N> div_down(tangents<double, N> a, tangents<double, N> b)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = div_down(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = div_down(fma_down(a.ds[i], b.v, -mul_down(a.v, b.ds[i])), mul_down(b.v, b.v));
    return res;
}

template<int N>
inline __device__ tangents<double, N> div_up(tangents<double, N> a, tangents<double, N> b)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = div_up(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = div_up(fma_up(a.ds[i], b.v, -mul_up(a.v, b.ds[i])), mul_up(b.v, b.v));
    return res;
}

template<int N>
inline __device__ tangents<double, N> median(tangents<double, N> a, tangents<double, N> b)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = div_up(a.v, b.v);
    int i = threadIdx.x;
    res.ds[i] = div_up(fma_up(a.ds[i], b.v, -mul_up(a.v, b.ds[i])), mul_up(b.v, b.v));
    return res;
}

// template<> inline __device__ tangent<double> median    (tangent<double> x, tangent<double> y) { return (x + y) * .5; }

template<int N>
inline __device__ tangents<double, N> min(tangents<double, N> x, tangents<double, N> y)
{ 
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = min(x.v, y.v);
    int i = threadIdx.x;
    res.ds[i] = min(x.ds[i], y.ds[i]);
    return res;
}

template<int N>
inline __device__ tangents<double, N> max(tangents<double, N> x, tangents<double, N> y)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = max(x.v, y.v);
    int i = threadIdx.x;
    res.ds[i] = max(x.ds[i], y.ds[i]);
    return res;
}

//     // template<> inline __device__ double copy_sign (double x, double y) { return copysign(x, y); }
//     template<> inline __device__ tangent<double> next_after(tangent<double> x, tangent<double> y) { using std::nextafter; return { nextafter(x.v, y.v), x.d }; }
//     template<> inline __device__ tangent<double> rcp_down  (tangent<double> x) { using std::pow; return { __drcp_rd(x.v), - __dmul_rd(pow(x.v, -2.0), x.d) }; }
//     template<> inline __device__ tangent<double> rcp_up    (tangent<double> x) { using std::pow; return { __drcp_ru(x.v), - __dmul_ru(pow(x.v, -2.0), x.d) }; }
//     template<> inline __device__ tangent<double> sqrt_down (tangent<double> x) { return sqrt(x); }
//     template<> inline __device__ tangent<double> sqrt_up   (tangent<double> x) { return sqrt(x); }
//     template<> inline __device__ tangent<double> int_down  (tangent<double> x) { return floor(x); }
//     template<> inline __device__ tangent<double> int_up    (tangent<double> x) { return ceil(x); }
//     // template<> inline __device__ double trunc     (double x)           { return ::trunc(x); }
//     // template<> inline __device__ double round_away(double x)           { return round(x); }
//     // template<> inline __device__ double round_even(double x)           { return nearbyint(x); }
//     template<> inline __device__ tangent<double> exp      (tangent<double> x) { return { ::exp(x.v), ::exp(x.d) }; }
//     // template<> inline __device__ double exp10     (double x)           { return ::exp10(x); }
//     // template<> inline __device__ double exp2      (double x)           { return ::exp2(x); }
//     template<> inline __device__ __host__ tangent<double> nan() { return { ::nan(""), ::nan("") }; }

// We need this dummy struct because C++ does not support partial function template specialization
template<typename T>
struct type{};

template<typename T, int N>
inline __device__ tangents<double, N> pos_inf(type<tangents<T, N>>)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = pos_inf<T>();
    int i = threadIdx.x;
    res.ds[i] = pos_inf<T>();
    return res;
}

template<typename T, int N>
inline __device__ tangents<double, N> neg_inf(type<tangents<T, N>>)
{
    CUTANGENT_SHARED tangents<double, N> res;
    res.v = neg_inf<T>();
    int i = threadIdx.x;
    res.ds[i] = neg_inf<T>();
    return res;
}

template<typename T>
inline __device__ T pos_inf()
{
    return pos_inf(type<T>{});
}

template<typename T>
inline __device__ T neg_inf()
{
    return neg_inf(type<T>{});
}

//     template<> inline __device__ tangent<double> next_floating(tangent<double> x) { return { nextafter(x.v, pos_inf<tangent<double>>().v), nextafter(x.d, pos_inf<tangent<double>>().d) }; }
//     template<> inline __device__ tangent<double> prev_floating(tangent<double> x) { return { nextafter(x.v, neg_inf<tangent<double>>().v), nextafter(x.d, neg_inf<tangent<double>>().d) }; }
//
// clang-format on
} // namespace cu::intrinsic

#undef CUTANGENT_SHARED

#pragma nv_diagnostic pop

#endif // CUTANGENT_ARITHMETIC_INTRINSIC_V_CUH
