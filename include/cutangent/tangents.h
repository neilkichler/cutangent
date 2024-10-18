#ifndef CUTANGENT_TANGENTS_H
#define CUTANGENT_TANGENTS_H

#include <compare>
#include <concepts>

#ifndef CUTANGENT_USE_SHARED_MEMORY
#define CUTANGENT_USE_SHARED_MEMORY 1
#endif

#ifndef CUTANGENT_USE_CONSERVATIVE_WARP_SYNC
#define CUTANGENT_USE_CONSERVATIVE_WARP_SYNC 1
#endif

// NOTE: Used in places where racecheck finds programming model hazards
//       but internal assumptions make it save to not use a sync
#if CUTANGENT_USE_CONSERVATIVE_WARP_SYNC
#define CUTANGENT_CONSERVATIVE_WARP_SYNC __syncwarp()
#else
#define CUTANGENT_CONSERVATIVE_WARP_SYNC
#endif

namespace cu
{

template<typename T, int N>
struct tangents
{
    T v;     // value
    T ds[N]; // derivatives

    constexpr tangents() = default;

    constexpr tangents(auto value)
        : v { static_cast<T>(value) }
    { }

    constexpr auto operator<=>(const tangents &other) const noexcept { return v <=> other.v; }
    constexpr bool operator==(const tangents &other) const noexcept { return v == other.v; }
};

template<typename T, int N>
constexpr T &value(tangents<T, N> &x)
{
    return x.v;
}

} // namespace cu

#endif // CUTANGENT_TANGENTS_H
