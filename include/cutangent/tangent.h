#ifndef CUTANGENT_TANGENT_H
#define CUTANGENT_TANGENT_H

#include <compare>

namespace cu
{

template<typename T>
struct tangent
{
    using value_type = T;

    T v; // value
    T d; // derivative

    constexpr tangent() = default;

    constexpr tangent(auto value)
        : v { static_cast<T>(value) }
        , d { static_cast<T>(0) }
    { }

    constexpr tangent(auto value, auto derivative)
        : v { value }
        , d { derivative }
    { }

    constexpr auto operator<=>(const tangent &other) const noexcept { return v <=> other.v; }
    constexpr bool operator==(const tangent &other) const noexcept { return v == other.v; }

    constexpr tangent &operator+=(const tangent &other)
    {
        v += other.v;
        d += other.d;
        return *this;
    }

    constexpr tangent &operator-=(const tangent &other)
    {
        v -= other.v;
        d -= other.d;
        return *this;
    }
};

template<typename T>
constexpr T &value(tangent<T> &x)
{
    return x.v;
}

template<typename T>
constexpr T &derivative(tangent<T> &x)
{
    return x.d;
}

} // namespace cu

#endif // CUTANGENT_TANGENT_H
