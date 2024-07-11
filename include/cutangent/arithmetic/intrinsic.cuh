#ifndef CUTANGENT_ARITHMETIC_INTRINSIC_CUH
#define CUTANGENT_ARITHMETIC_INTRINSIC_CUH

#include <cutangent/arithmetic/basic.cuh>
#include <cutangent/tangent.h>

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

    template<> inline __device__ tangent<double> fma_down  (tangent<double> x, tangent<double> y, tangent<double> z) { return x * y + z; }
    template<> inline __device__ tangent<double> fma_up    (tangent<double> x, tangent<double> y, tangent<double> z) { return x * y + z; }
    template<> inline __device__ tangent<double> add_down  (tangent<double> x, tangent<double> y) { return x + y; }
    template<> inline __device__ tangent<double> add_up    (tangent<double> x, tangent<double> y) { return x + y; }
    template<> inline __device__ tangent<double> sub_down  (tangent<double> x, tangent<double> y) { return x - y; }
    template<> inline __device__ tangent<double> sub_up    (tangent<double> x, tangent<double> y) { return x - y; }
    template<> inline __device__ tangent<double> mul_down  (tangent<double> x, tangent<double> y) { return x * y; }
    template<> inline __device__ tangent<double> mul_up    (tangent<double> x, tangent<double> y) { return x * y; }
    template<> inline __device__ tangent<double> div_down  (tangent<double> x, tangent<double> y) { return x / y; }
    template<> inline __device__ tangent<double> div_up    (tangent<double> x, tangent<double> y) { return x / y; }
    template<> inline __device__ tangent<double> median    (tangent<double> x, tangent<double> y) { return (x + y) * .5; }
    // template<> inline __device__ double min       (double x, double y) { return fmin(x, y); }
    // template<> inline __device__ double max       (double x, double y) { return fmax(x, y); }
    // template<> inline __device__ double copy_sign (double x, double y) { return copysign(x, y); }
    template<> inline __device__ tangent<double> next_after(tangent<double> x, tangent<double> y) { using std::nextafter; return { nextafter(x.v, y.v), x.d }; }
    template<> inline __device__ tangent<double> rcp_down  (tangent<double> x) { using std::pow; return { __drcp_rd(x.v), - __dmul_rd(pow(x.v, -2.0), x.d) }; }
    template<> inline __device__ tangent<double> rcp_up    (tangent<double> x) { using std::pow; return { __drcp_ru(x.v), - __dmul_ru(pow(x.v, -2.0), x.d) }; }
    template<> inline __device__ tangent<double> sqrt_down (tangent<double> x) { return sqrt(x); }
    template<> inline __device__ tangent<double> sqrt_up   (tangent<double> x) { return sqrt(x); }
    template<> inline __device__ tangent<double> int_down  (tangent<double> x) { return floor(x); }
    template<> inline __device__ tangent<double> int_up    (tangent<double> x) { return ceil(x); }
    // template<> inline __device__ double trunc     (double x)           { return ::trunc(x); }
    // template<> inline __device__ double round_away(double x)           { return round(x); }
    // template<> inline __device__ double round_even(double x)           { return nearbyint(x); }
    template<> inline __device__ tangent<double> exp      (tangent<double> x) { return { ::exp(x.v), ::exp(x.d) }; }
    // template<> inline __device__ double exp10     (double x)           { return ::exp10(x); }
    // template<> inline __device__ double exp2      (double x)           { return ::exp2(x); }
    template<> inline __device__ __host__ tangent<double> nan() { return { ::nan(""), ::nan("") }; }
    template<> inline __device__ tangent<double> neg_inf() { return { __longlong_as_double(0xfff0000000000000ull), 0.0 }; }
    template<> inline __device__ tangent<double> pos_inf() { return { __longlong_as_double(0x7ff0000000000000ull), 0.0 }; }
    template<> inline __device__ tangent<double> next_floating(tangent<double> x) { return { nextafter(x.v, pos_inf<tangent<double>>().v), 0.0 }; }
    template<> inline __device__ tangent<double> prev_floating(tangent<double> x) { return { nextafter(x.v, neg_inf<tangent<double>>().v), 0.0 }; }

// clang-format on
} // namespace cu::intrinsic

#endif // CUTANGENT_ARITHMETIC_INTRINSIC_CUH
