#ifndef CUTANGENT_ARITHMETIC_INTRINSIC_CUH
#define CUTANGENT_ARITHMETIC_INTRINSIC_CUH

#include <cutangent/arithmetic/basic.cuh>
#include <cutangent/arithmetic/intrinsics_api.h>
#include <cutangent/tangent.h>
#include <cutangent/traits.h>

#include <limits>

namespace cu
{
// clang-format off

template<numeric T>
struct intrinsics<tangent<T>>
{
    // required to avoid recursive template instantiation attempt since, e.g., double is convertible back to tangent<double>.
    using i = intrinsics<T>;

    using X = tangent<T>;
    using Y = tangent<T>;
    using Z = tangent<T>;

    static __device__ tangent<T> fma_down     (X x, Y y, Z z) { return { i::fma_down(x.v, y.v, z.v), i::fma_down(x.v, y.d, i::fma_down(x.d, y.v, z.d)) }; }
    static __device__ tangent<T> fma_up       (X x, Y y, Z z) { return { i::fma_up  (x.v, y.v, z.v), i::fma_up  (x.v, y.d, i::fma_up  (x.d, y.v, z.d)) }; }
    static __device__ tangent<T> add_down     (X x, Y y) { return { i::add_down(x.v, y.v), i::add_down(x.d, y.d) }; }
    static __device__ tangent<T> add_up       (X x, Y y) { return { i::add_up  (x.v, y.v), i::add_up  (x.d, y.d) }; }
    static __device__ tangent<T> sub_down     (X x, Y y) { return { i::sub_down(x.v, y.v), i::sub_down(x.d, y.d) }; }
    static __device__ tangent<T> sub_up       (X x, Y y) { return { i::sub_up  (x.v, y.v), i::sub_up  (x.d, y.d) }; }
    static __device__ tangent<T> mul_down     (X x, Y y) { return { i::mul_down(x.v, y.v), i::fma_down(x.v, y.d, i::mul_down(x.d, y.v)) }; }
    static __device__ tangent<T> mul_up       (X x, Y y) { return { i::mul_up  (x.v, y.v), i::fma_up  (x.v, y.d, i::mul_up  (x.d, y.v)) }; }
    static __device__ tangent<T> div_down     (X x, Y y) { return { i::div_down(x.v, y.v), i::div_down(i::fma_down(x.d, y.v, -i::mul_up  (x.v, y.d)), i::mul_up  (y.v, y.v)) }; }
    static __device__ tangent<T> div_up       (X x, Y y) { return { i::div_up  (x.v, y.v), i::div_up  (i::fma_up  (x.d, y.v, -i::mul_down(x.v, y.d)), i::mul_down(y.v, y.v)) }; }
    static __device__ tangent<T> median       (X x, Y y) { return (x + y) * .5; }
    static __device__ tangent<T> min          (X x, Y y) { return { i::min(x.v, y.v), i::min(x.d, y.d) }; }
    static __device__ tangent<T> max          (X x, Y y) { return { i::max(x.v, y.v), i::max(x.d, y.d) }; }
    static __device__ tangent<T> next_after   (X x, Y y) { return { i::next_after(x.v, y.v), x.d }; }
    static __device__ tangent<T> round_towards(X x, tangent<T> to, unsigned int n) { return { i::round_towards(x.v, to.v, n), x.d }; }
    static __device__ tangent<T> rcp_down     (X x) { return { i::rcp_down(x.v), -i::mul_down(i::rcp_down(i::mul_down(x.v, x.v)), x.d) }; }
    static __device__ tangent<T> rcp_up       (X x) { return { i::rcp_up  (x.v), -i::mul_up  (i::rcp_up  (i::mul_up  (x.v, x.v)), x.d) }; }
    static __device__ tangent<T> sqrt_down    (X x) { auto sqrt_x_v = i::sqrt_down(x.v); return { sqrt_x_v, i::div_down(x.d, i::add_up  (T{2} * sqrt_x_v, x.v == T{} ? std::numeric_limits<T>::min() : T{})) }; }
    static __device__ tangent<T> sqrt_up      (X x) { auto sqrt_x_v = i::sqrt_up  (x.v); return { sqrt_x_v, i::div_up  (x.d, i::add_down(T{2} * sqrt_x_v, x.v == T{} ? std::numeric_limits<T>::min() : T{})) }; }
    static __device__ tangent<T> int_down     (X x) { return floor(x); }
    static __device__ tangent<T> int_up       (X x) { return ceil(x); }
    static __device__ tangent<T> round_away   (X x) { return round(x); }
    static __device__ tangent<T> round_even   (X x) { return nearbyint(x); }
    static __device__ tangent<T> next_floating(X x) { return { nextafter(x.v, +std::numeric_limits<T>::infinity()), nextafter(x.d, std::numeric_limits<T>::infinity()) }; }
    static __device__ tangent<T> prev_floating(X x) { return { nextafter(x.v, -std::numeric_limits<T>::infinity()), nextafter(x.d, std::numeric_limits<T>::infinity()) }; }
};

template<numeric T>
struct intrinsics<tangent<T>, T>
{
    using i = intrinsics<T>;

    static __device__ tangent<T> add_down(tangent<T> x, T y) { return { i::add_down(x.v, y), x.d }; }
    static __device__ tangent<T> add_up  (tangent<T> x, T y) { return { i::add_up  (x.v, y), x.d }; }
    static __device__ tangent<T> sub_down(tangent<T> x, T y) { return { i::sub_down(x.v, y), x.d }; }
    static __device__ tangent<T> sub_up  (tangent<T> x, T y) { return { i::sub_up  (x.v, y), x.d }; }
    static __device__ tangent<T> mul_down(tangent<T> x, T y) { return { i::mul_down(x.v, y), i::mul_down(x.d, y) }; }
    static __device__ tangent<T> mul_up  (tangent<T> x, T y) { return { i::mul_up  (x.v, y), i::mul_up  (x.d, y) }; }
    static __device__ tangent<T> div_down(tangent<T> x, T y) { return { i::div_down(x.v, y), i::div_down(x.d, y) }; }
    static __device__ tangent<T> div_up  (tangent<T> x, T y) { return { i::div_up  (x.v, y), i::div_up  (x.d, y) }; }
};

template<numeric T>
struct intrinsics<T, tangent<T>>
{
    using i = intrinsics<T>;

    static __device__ tangent<T> add_down(T x, tangent<T> y) { return { i::add_down(x, y.v), y.d }; }
    static __device__ tangent<T> add_up  (T x, tangent<T> y) { return { i::add_up  (x, y.v), y.d }; }
    static __device__ tangent<T> sub_down(T x, tangent<T> y) { return { i::sub_down(x, y.v), y.d }; }
    static __device__ tangent<T> sub_up  (T x, tangent<T> y) { return { i::sub_up  (x, y.v), y.d }; }
    static __device__ tangent<T> mul_down(T x, tangent<T> y) { return { i::mul_down(x, y.v), i::mul_down(x, y.d) }; }
    static __device__ tangent<T> mul_up  (T x, tangent<T> y) { return { i::mul_up  (x, y.v), i::mul_up  (x, y.d) }; }
    static __device__ tangent<T> div_down(T x, tangent<T> y) { return { i::div_down(x, y.v), i::div_down(x, y.d) }; }
    static __device__ tangent<T> div_up  (T x, tangent<T> y) { return { i::div_up  (x, y.v), i::div_up  (x, y.d) }; }
};

// clang-format on
}; // namespace cu

#endif // CUTANGENT_ARITHMETIC_INTRINSIC_CUH
