#ifndef CUTANGENT_TANGENT_H
#define CUTANGENT_TANGENT_H

#include <compare>

namespace cu
{

template<typename T>
struct tangent
{
    T v; // value
    T d; // derivative

    constexpr auto operator<=>(const tangent &) const = default;
};


} // namespace cu

#endif // CUTANGENT_TANGENT_H
