#ifndef CUTANGENT_FORMAT_H
#define CUTANGENT_FORMAT_H

#include <cutangent/tangent.h>

#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, tangent<T> x)
{
    return os << "{v: " << x.v << ", d: " << x.d << "}";
}

} // namespace cu

#endif // CUTANGENT_FORMAT_H
