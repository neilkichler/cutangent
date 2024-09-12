#ifndef CUTANGENT_ARITHMETIC_INTRINSIC_V_CUH
#define CUTANGENT_ARITHMETIC_INTRINSIC_V_CUH

#include <cutangent/tangents.h>

#pragma nv_diagnostic push
// ignore "821: function was referenced but not defined" for e.g. add_down since
// this has to be defined by the type and is only included in an upstream project.
#pragma nv_diag_suppress 821

namespace cu::intrinsic
{
// clang-format off
//     template<typename T> inline __device__ T fma_down  (T x, T y, T z);
//     template<typename T> inline __device__ T fma_up    (T x, T y, T z);
template<typename T> inline __device__ T add_down(T x, T y);
template<typename T> inline __device__ T add_up(T x, T y);
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
//
//     template<> inline __device__ tangent<double> fma_down  (tangent<double> x, tangent<double> y, tangent<double> z) { return x * y + z; }
//     template<> inline __device__ tangent<double> fma_up    (tangent<double> x, tangent<double> y, tangent<double> z) { return x * y + z; }
template<int N>
inline __device__ tangents<double, N> add_down(tangents<double, N> x, tangents<double, N> y)
{
    tangents<double, N> res;
    res.v = add_down(x.v, y.v);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = add_down(x.ds[i], y.ds[i]);
    }
    return res;
}

template<int N>
inline __device__ tangents<double, N> add_up(tangents<double, N> a, tangents<double, N> b)
{
    tangents<double, N> res;
    res.v = add_up(a.v, b.v);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = add_up(a.ds[i], b.ds[i]);
    }
    return res;
}

template<int N>
inline __device__ tangents<double, N> sub_down(tangents<double, N> a, tangents<double, N> b)
{ 
    tangents<double, N> res;
    res.v = sub_down(a.v, b.v);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = sub_down(a.ds[i], b.ds[i]);
    }
    return res;
}

template<int N>
inline __device__ tangents<double, N> sub_up(tangents<double, N> a, tangents<double, N> b)
{
    tangents<double, N> res;
    res.v = sub_up(a.v, b.v);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = sub_up(a.ds[i], b.ds[i]);
    }
    return res;
}

template<int N>
inline __device__ tangents<double, N> mul_down(tangents<double, N> a, tangents<double, N> b)
{
    tangents<double, N> res;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        printf("a.ds[%d] = %g\n", i, a.ds[i]);
        printf("b.ds[%d] = %g\n", i, b.ds[i]);
        res.ds[i] = add_down(mul_down(a.v, b.ds[i]), mul_down(a.ds[i], b.v));
        printf("res.ds[%d] = %g\n", i, res.ds[i]);
    }
    res.v = mul_down(a.v, b.v);
    return res;
}

template<int N>
inline __device__ tangents<double, N> mul_up(tangents<double, N> a, tangents<double, N> b)
{
    tangents<double, N> res;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = add_up(mul_up(a.v, b.ds[i]), mul_up(a.ds[i], b.v));
    }
    res.v = mul_up(a.v, b.v);
    return res;
}
//     template<> inline __device__ tangent<double> div_down  (tangent<double> x, tangent<double> y) { return x / y; }
//     template<> inline __device__ tangent<double> div_up    (tangent<double> x, tangent<double> y) { return x / y; }
//     template<> inline __device__ tangent<double> median    (tangent<double> x, tangent<double> y) { return (x + y) * .5; }
template<int N>
inline __device__ tangents<double, N> min (tangents<double, N> x, tangents<double, N> y)
{ 
    return min(x, y);
}

template<int N>
inline __device__ tangents<double, N> max (tangents<double, N> x, tangents<double, N> y)
{
    return max(x, y);
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
    tangents<double, N> res;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = pos_inf<T>();
    }
    res.v = pos_inf<T>();
    return res;
}

template<typename T, int N>
inline __device__ tangents<double, N> neg_inf(type<tangents<T, N>>)
{
    tangents<double, N> res;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        res.ds[i] = neg_inf<T>();
    }
    res.v = neg_inf<T>();
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

#pragma nv_diagnostic pop

#endif // CUTANGENT_ARITHMETIC_INTRINSIC_V_CUH
