#ifndef CUTANGENT_ARITHMETIC_INTRINSIC_CUH
#define CUTANGENT_ARITHMETIC_INTRINSIC_CUH

#include <cutangent/arithmetic/basic.cuh>
#include <cutangent/tangent.h>

#include <limits>

#pragma nv_diagnostic push
// ignore "821: function was referenced but not defined" for e.g. add_down since
// this has to be defined by the type and is only included in an upstream project.
#pragma nv_diag_suppress 821

namespace cu::intrinsic
{
// clang-format off
    template<typename T> inline __device__ T fma_down  (T x, T y, T z);
    template<typename T> inline __device__ T fma_up    (T x, T y, T z);
    template<typename T> inline __device__ T add_down  (T x, T y);
    template<typename T> inline __device__ T add_up    (T x, T y);
    template<typename T> inline __device__ T sub_down  (T x, T y);
    template<typename T> inline __device__ T sub_up    (T x, T y);
    template<typename T> inline __device__ T mul_down  (T x, T y);
    template<typename T> inline __device__ T mul_up    (T x, T y);
    template<typename T> inline __device__ T div_down  (T x, T y);
    template<typename T> inline __device__ T div_up    (T x, T y);
    template<typename T> inline __device__ T median    (T x, T y);
    template<typename T> inline __device__ T min       (T x, T y);
    template<typename T> inline __device__ T max       (T x, T y);
    template<typename T> inline __device__ T copy_sign (T x, T y);
    template<typename T, typename U> inline __device__ T next_after(T x, U y);
    template<typename T> inline __device__ T rcp_down  (T x);
    template<typename T> inline __device__ T rcp_up    (T x);
    template<typename T> inline __device__ T sqrt_down (T x);
    template<typename T> inline __device__ T sqrt_up   (T x);
    template<typename T> inline __device__ T int_down  (T x);
    template<typename T> inline __device__ T int_up    (T x);
    template<typename T> inline __device__ T trunc     (T x);
    template<typename T> inline __device__ T round_away(T x);
    template<typename T> inline __device__ T round_even(T x);
    template<typename T> inline __device__ T exp       (T x);
    template<typename T> inline __device__ T exp10     (T x);
    template<typename T> inline __device__ T exp2      (T x);
    template<typename T> inline __device__ __host__ T nan();
    template<typename T> inline __device__ T pos_inf();
    template<typename T> inline __device__ T neg_inf();
    template<typename T> inline __device__ T next_floating(T x);
    template<typename T> inline __device__ T prev_floating(T x);

    using cu::tangent;

    template<> inline __device__ tangent<double> fma_down  (tangent<double> x, tangent<double> y, tangent<double> z) { return { fma_down(x.v, y.v, z.v), fma_down(x.v, y.d, fma_down(x.d, y.v, z.d)) }; }
    template<> inline __device__ tangent<double> fma_up    (tangent<double> x, tangent<double> y, tangent<double> z) { return { fma_up  (x.v, y.v, z.v), fma_up  (x.v, y.d, fma_up  (x.d, y.v, z.d)) }; }
    template<> inline __device__ tangent<double> add_down  (tangent<double> x, tangent<double> y)                    { return { add_down(x.v, y.v), add_down(x.d, y.d) }; }
    template<> inline __device__ tangent<double> add_up    (tangent<double> x, tangent<double> y)                    { return { add_up  (x.v, y.v), add_up  (x.d, y.d) }; }
    template<> inline __device__ tangent<double> sub_down  (tangent<double> x, tangent<double> y)                    { return { sub_down(x.v, y.v), sub_down(x.d, y.d) }; }
    template<> inline __device__ tangent<double> sub_up    (tangent<double> x, tangent<double> y)                    { return { sub_up  (x.v, y.v), sub_up  (x.d, y.d) }; }
    template<> inline __device__ tangent<double> mul_down  (tangent<double> x, tangent<double> y)                    { return { mul_down(x.v, y.v), fma_down(x.v, y.d, mul_down(x.d, y.v)) }; }
    template<> inline __device__ tangent<double> mul_up    (tangent<double> x, tangent<double> y)                    { return { mul_up  (x.v, y.v), fma_up  (x.v, y.d, mul_up  (x.d, y.v)) }; }
    template<> inline __device__ tangent<double> div_down  (tangent<double> x, tangent<double> y)                    { return { div_down(x.v, y.v), div_down(fma_down(x.d, y.v, -mul_up  (x.v, y.d)), mul_up  (y.v, y.v)) }; }
    template<> inline __device__ tangent<double> div_up    (tangent<double> x, tangent<double> y)                    { return { div_up  (x.v, y.v), div_up  (fma_up  (x.d, y.v, -mul_down(x.v, y.d)), mul_down(y.v, y.v)) }; }
    template<> inline __device__ tangent<double> median    (tangent<double> x, tangent<double> y)                    { return (x + y) * .5; }
    template<> inline __device__ tangent<double> min       (tangent<double> x, tangent<double> y)                    { return min(x, y); }
    template<> inline __device__ tangent<double> max       (tangent<double> x, tangent<double> y)                    { return max(x, y); }
    // template<> inline __device__ double copy_sign (double x, double y) { return copysign(x, y); }
    template<> inline __device__ tangent<double> next_after(tangent<double> x, tangent<double> y) { using std::nextafter; return { nextafter(x.v, y.v), x.d }; }
    template<> inline __device__ tangent<double> rcp_down  (tangent<double> x) { using std::pow; return { __drcp_rd(x.v), - __dmul_rd(pow(x.v, -2.0), x.d) }; }
    template<> inline __device__ tangent<double> rcp_up    (tangent<double> x) { using std::pow; return { __drcp_ru(x.v), - __dmul_ru(pow(x.v, -2.0), x.d) }; }
    template<> inline __device__ tangent<double> sqrt_down (tangent<double> x) { auto sqrt_x_v = sqrt_down(x.v); return { sqrt_x_v, div_down(x.d, add_up  (2.0 * sqrt_x_v, x.v == 0.0 ? std::numeric_limits<double>::min() : 0.0)) }; }
    template<> inline __device__ tangent<double> sqrt_up   (tangent<double> x) { auto sqrt_x_v = sqrt_up  (x.v); return { sqrt_x_v, div_up  (x.d, add_down(2.0 * sqrt_x_v, x.v == 0.0 ? std::numeric_limits<double>::min() : 0.0)) }; }
    template<> inline __device__ tangent<double> int_down  (tangent<double> x) { return floor(x); }
    template<> inline __device__ tangent<double> int_up    (tangent<double> x) { return ceil(x); }
    // template<> inline __device__ double trunc     (double x)           { return ::trunc(x); }
    // template<> inline __device__ double round_away(double x)           { return round(x); }
    // template<> inline __device__ double round_even(double x)           { return nearbyint(x); }
    template<> inline __device__ tangent<double> exp      (tangent<double> x) { return { ::exp(x.v), ::exp(x.d) }; }
    // template<> inline __device__ double exp10     (double x)           { return ::exp10(x); }
    // template<> inline __device__ double exp2      (double x)           { return ::exp2(x); }
    template<> inline __device__ __host__ tangent<double> nan() { return { ::nan(""), ::nan("") }; }
    template<> inline __device__ tangent<double> neg_inf() { return { __longlong_as_double(0xfff0000000000000ull), __longlong_as_double(0xfff0000000000000ull) }; }
    template<> inline __device__ tangent<double> pos_inf() { return { __longlong_as_double(0x7ff0000000000000ull), __longlong_as_double(0x7ff0000000000000ull) }; }
    template<> inline __device__ tangent<double> next_floating(tangent<double> x) { return { nextafter(x.v, pos_inf<tangent<double>>().v), nextafter(x.d, pos_inf<tangent<double>>().d) }; }
    template<> inline __device__ tangent<double> prev_floating(tangent<double> x) { return { nextafter(x.v, neg_inf<tangent<double>>().v), nextafter(x.d, neg_inf<tangent<double>>().d) }; }

#define fn(T) template<typename T> inline constexpr __device__ auto

    fn(T) add_down(const T &x, typename T::value_type y) -> T { return { add_down(x.v, y), x.d }; }
    fn(T) add_down(typename T::value_type x, const T &y) -> T { return { add_down(x, y.v), y.d }; }
    fn(T) add_up  (const T &x, typename T::value_type y) -> T { return { add_up  (x.v, y), x.d }; }
    fn(T) add_up  (typename T::value_type x, const T &y) -> T { return { add_up  (x, y.v), y.d }; }
    fn(T) sub_down(const T &x, typename T::value_type y) -> T { return { sub_down(x.v, y), x.d }; }
    fn(T) sub_down(typename T::value_type x, const T &y) -> T { return { sub_down(x, y.v), y.d }; }
    fn(T) sub_up  (const T &x, typename T::value_type y) -> T { return { sub_up  (x.v, y), x.d }; }
    fn(T) sub_up  (typename T::value_type x, const T &y) -> T { return { sub_up  (x, y.v), y.d }; }
    fn(T) mul_down(const T &x, typename T::value_type y) -> T { return { mul_down(x.v, y), mul_down(x.d, y) }; }
    fn(T) mul_down(typename T::value_type x, const T &y) -> T { return { mul_down(x, y.v), mul_down(x, y.d) }; }
    fn(T) mul_up  (const T &x, typename T::value_type y) -> T { return { mul_up  (x.v, y), mul_up  (x.d, y) }; }
    fn(T) mul_up  (typename T::value_type x, const T &y) -> T { return { mul_up  (x, y.v), mul_up  (x, y.d) }; }

#undef fn

// clang-format on
} // namespace cu::intrinsic

#pragma nv_diagnostic pop

#endif // CUTANGENT_ARITHMETIC_INTRINSIC_CUH
