#ifndef CUTANGENT_FORMAT_H
#define CUTANGENT_FORMAT_H

#include <cutangent/tangent.h>
#include <cutangent/vtangent.h>

#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, tangent<T> x)
{
    return os << "{v: " << x.v << ", d: " << x.d << "}";
}

template<typename T, int N>
std::ostream &operator<<(std::ostream &os, vtangent<T, N> x)
{
    for (int i = 0; i < N; ++i)
        return os << "{v: " << x.v << ", d: " << x.ds[i] << "}";

    return os;
}

} // namespace cu

#endif // CUTANGENT_FORMAT_H
