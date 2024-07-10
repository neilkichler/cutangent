#ifndef CUTANGENT_ARITHMETIC_BASIC_CUH
#define CUTANGENT_ARITHMETIC_BASIC_CUH

#include <cutangent/tangent.h>

#include <algorithm>
#include <cmath>
#include <concepts>

namespace cu
{

#define fn inline constexpr __device__

template<typename T>
fn tangent<T> operator-(tangent<T> x)
{
    return { -x.v, -x.d };
}

template<typename T>
fn tangent<T> operator+(tangent<T> a, tangent<T> b)
{
    return { a.v + b.v, a.d + b.d };
}

template<typename T>
fn tangent<T> operator-(tangent<T> a, tangent<T> b)
{
    return { a.v - b.v, a.d - b.d };
}

template<typename T>
fn tangent<T> operator*(tangent<T> a, tangent<T> b)
{
    return { a.v * b.v, a.v * b.d + a.d * b.v };
}

template<typename T>
fn tangent<T> operator*(tangent<T> a, T b)
{
    return { a.v * b, a.d * b };
}

template<typename T>
fn tangent<T> operator*(T a, tangent<T> b)
{
    return { a * b.v, a * b.d };
}

template<typename T>
fn tangent<T> operator/(tangent<T> a, tangent<T> b)
{
    return { a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v) };
}

template<typename T>
fn tangent<T> max(tangent<T> a, tangent<T> b)
{
    using std::max;

    return { max(a.v, b.v),
             a.v >= b.v ? a.d : b.d }; // '>=' instead of '>' due to subgradient
}

template<typename T>
fn tangent<T> min(tangent<T> a, tangent<T> b)
{
    using std::min;

    return { min(a.v, b.v),
             a.v <= b.v ? a.d : b.d }; // '<=' instead of '<' due to subgradient
}

template<typename T>
fn tangent<T> abs(tangent<T> x)
{
    using std::abs;
    using std::copysign;
    // NOTE: not differentiable at x = 0.
    return { abs(x.v), copysign(1.0, x.v) * x.d };
}

template<typename T>
fn tangent<T> clamp(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    using std::clamp;

    return { clamp(v.v, lb.v, ub.v), lb.d * (v.v < lb.v) + ub.d * (v.v > ub.v) };
}

template<typename T>
fn tangent<T> mid(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    return clamp(v, lb, ub);
}

template<typename T>
fn tangent<T> sin(tangent<T> x)
{
    using std::cos;
    using std::sin;

    return { sin(x.v), cos(x.v) * x.d };
}

template<typename T>
fn tangent<T> cos(tangent<T> x)
{
    using std::cos;
    using std::sin;

    return { cos(x.v), -sin(x.v) * x.d };
}

template<typename T>
fn tangent<T> exp(tangent<T> x)
{
    using std::exp;

    return { exp(x.v), exp(x.v) * x.d };
}

template<typename T>
fn tangent<T> log(tangent<T> x)
{
    using std::log;

    // NOTE: We currently do not treat the case where x.v == 0, x.d != 0 to map to -+inf.
    return { log(x.v), x.d / x.v };
}

template<typename T>
fn tangent<T> sqr(tangent<T> x)
{
    return { sqr(x.v), 2.0 * x.v * x.d };
}

template<typename T>
fn tangent<T> sqrt(tangent<T> x)
{
    using std::sqrt;
    // NOTE: We currently do not treat the case where x.v == 0, x.d > 0 to map to +inf.
    return { sqrt(x.v), x.d / (2.0 * sqrt(x.v)) };
}

template<typename T>
fn tangent<T> pown(tangent<T> x, auto n)
{
    using std::pow;

    return { pow(x.v, n), n * pow(x.v, n - 1) * x.d };
}

template<typename T>
fn tangent<T> pown(auto x, tangent<T> n)
{
    using std::pow;

    return { pow(x, n.v), pow(x, n.v) * log(x) * n.d };
}

template<typename T>
fn tangent<T> pow(tangent<T> x, auto n)
{
    return pown(x, n);
}

template<typename T>
fn tangent<T> pow(auto x, tangent<T> n)
{
    return pown(x, n);
}

template<typename T>
fn tangent<T> pow(tangent<T> x, tangent<T> n)
{
    using std::pow;

    return { pow(x.v, n.v),
             n.v * pow(x.v, n.v - 1) * x.d + pow(x.v, n.v) * log(x.v) * n.d };
}

template<typename T>
fn bool isinf(tangent<T> a)
{
    using std::isinf;

    return isinf(a.v);
}

#undef fn

namespace intrinsic
{

    // template<typename T> inline __device__ T fma_down  (T x, T y, T z);
    // template<typename T> inline __device__ T fma_up    (T x, T y, T z);
    template<typename T>
    inline __device__ T add_down(T x, T y);
    template<typename T>
    inline __device__ T add_up(T x, T y);
    template<typename T>
    inline __device__ T sub_down(T x, T y);
    template<typename T>
    inline __device__ T sub_up(T x, T y);
    template<typename T>
    inline __device__ T mul_down(T x, T y);
    template<typename T>
    inline __device__ T mul_up(T x, T y);
    template<typename T>
    inline __device__ T div_down(T x, T y);
    template<typename T>
    inline __device__ T div_up(T x, T y);
    // template<typename T> inline __device__ T median    (T x, T y);
    // template<typename T> inline __device__ T min       (T x, T y);
    // template<typename T> inline __device__ T max       (T x, T y);
    // template<typename T> inline __device__ T copy_sign (T x, T y);
    template<typename T>
    inline __device__ T next_after(T x, T y);
    // template<typename T> inline __device__ T rcp_down  (T x);
    // template<typename T> inline __device__ T rcp_up    (T x);
    template<typename T>
    inline __device__ T sqrt_down(T x);
    template<typename T>
    inline __device__ T sqrt_up(T x);
    // template<typename T> inline __device__ T int_down  (T x);
    // template<typename T> inline __device__ T int_up    (T x);
    // template<typename T> inline __device__ T trunc     (T x);
    // template<typename T> inline __device__ T round_away(T x);
    // template<typename T> inline __device__ T round_even(T x);
    template<typename T>
    inline __device__ T exp(T x);
    // template<typename T> inline __device__ T exp10     (T x);
    // template<typename T> inline __device__ T exp2      (T x);
    template<typename T>
    inline __device__ __host__ T nan();
    // template<typename T> inline __device__ T pos_inf();
    // template<typename T> inline __device__ T neg_inf();
    // template<typename T> inline __device__ T next_floating(T x);
    // template<typename T> inline __device__ T prev_floating(T x);

    // template<> inline __device__ double fma_down  (double x, double y, double z) { return __fma_rd(x, y, z); }
    // template<> inline __device__ double fma_up    (double x, double y, double z) { return __fma_ru(x, y, z); }
    template<>
    inline __device__ cu::tangent<double> add_down(cu::tangent<double> x, cu::tangent<double> y) { return (x + y); }
    template<>
    inline __device__ cu::tangent<double> add_up(cu::tangent<double> x, cu::tangent<double> y) { return (x + y); }
    template<>
    inline __device__ cu::tangent<double> sub_down(cu::tangent<double> x, cu::tangent<double> y) { return (x - y); }
    template<>
    inline __device__ cu::tangent<double> sub_up(cu::tangent<double> x, cu::tangent<double> y) { return (x - y); }
    template<>
    inline __device__ cu::tangent<double> mul_down(cu::tangent<double> x, cu::tangent<double> y) { return (x * y); }
    template<>
    inline __device__ cu::tangent<double> mul_up(cu::tangent<double> x, cu::tangent<double> y) { return (x * y); }
    template<>
    inline __device__ cu::tangent<double> div_down(cu::tangent<double> x, cu::tangent<double> y) { return (x / y); }
    template<>
    inline __device__ cu::tangent<double> div_up(cu::tangent<double> x, cu::tangent<double> y) { return (x / y); }
    // template<> inline __device__ double median    (double x, double y) { return (x + y) * .5; }
    // template<> inline __device__ double min       (double x, double y) { return fmin(x, y); }
    // template<> inline __device__ double max       (double x, double y) { return fmax(x, y); }
    // template<> inline __device__ double copy_sign (double x, double y) { return copysign(x, y); }
    template<>
    inline __device__ cu::tangent<double> next_after(cu::tangent<double> x, cu::tangent<double> y) { return { nextafter(x.v, y.v), x.d }; }
    // template<> inline __device__ double rcp_down  (double x)           { return __drcp_rd(x); }
    // template<> inline __device__ double rcp_up    (double x)           { return __drcp_ru(x); }
    template<>
    inline __device__ cu::tangent<double> sqrt_down(cu::tangent<double> x) { return sqrt(x); }
    template<>
    inline __device__ cu::tangent<double> sqrt_up(cu::tangent<double> x) { return sqrt(x); }
    // template<> inline __device__ double int_down  (double x)           { return floor(x); }
    // template<> inline __device__ double int_up    (double x)           { return ceil(x); }
    // template<> inline __device__ double trunc     (double x)           { return ::trunc(x); }
    // template<> inline __device__ double round_away(double x)           { return round(x); }
    // template<> inline __device__ double round_even(double x)           { return nearbyint(x); }
    template<>
    inline __device__ cu::tangent<double> exp(cu::tangent<double> x) { return { ::exp(x.v), ::exp(x.d) }; }
    // template<> inline __device__ double exp10     (double x)           { return ::exp10(x); }
    // template<> inline __device__ double exp2      (double x)           { return ::exp2(x); }
    template<>
    inline __device__ __host__ cu::tangent<double> nan() { return { ::nan(""), ::nan("") }; }
    // template<> inline __device__ double neg_inf() { return __longlong_as_double(0xfff0000000000000ull); }
    // template<> inline __device__ double pos_inf() { return __longlong_as_double(0x7ff0000000000000ull); }
    // template<> inline __device__ double next_floating(double x)        { return nextafter(x, intrinsic::pos_inf<double>()); }
    // template<> inline __device__ double prev_floating(double x)        { return nextafter(x, intrinsic::neg_inf<double>()); }

    template<typename T>
    __device__ T pos_inf();
    template<typename T>
    __device__ T neg_inf();
    template<typename T>
    __device__ T next_floating(T x);
    template<typename T>
    __device__ T prev_floating(T x);

    template<>
    __device__ cu::tangent<double> neg_inf()
    {
        return cu::tangent<double> { __longlong_as_double(0xfff0000000000000ull), 0.0 };
    }

    template<>
    __device__ cu::tangent<double> pos_inf()
    {
        return { __longlong_as_double(0x7ff0000000000000ull), 0.0 };
    }

    template<>
    __device__ cu::tangent<double> next_floating(cu::tangent<double> x)
    {
        return cu::tangent<double> { nextafter(x.v, pos_inf<double>()), 0.0 };
    }

    template<>
    __device__ cu::tangent<double> prev_floating(cu::tangent<double> x)
    {
        return cu::tangent<double> { nextafter(x.v, neg_inf<double>()), 0.0 };
    }
} // namespace intrinsic

} // namespace cu

#endif // CUTANGENT_ARITHMETIC_BASIC_CUH
