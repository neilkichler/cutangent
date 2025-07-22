<h1 align='center'>CuTangent

[![Cpp Version](https://img.shields.io/badge/requires-C++20-blue)](https://github.com/neilkichler/cuinterval/tree/main?tab=readme-ov-file#build-requirements)
[![CUDA Version](https://img.shields.io/badge/CUDA-12+-8A2BE2?logo=nvidia)](https://github.com/neilkichler/cuinterval/tree/main?tab=readme-ov-file#build-requirements)
[![CMake Version](https://img.shields.io/badge/CMake-3.25.2+-blue?logo=cmake)](https://github.com/neilkichler/cuinterval/tree/main?tab=readme-ov-file#build-requirements)
[![Docs](https://img.shields.io/badge/documentation-latest-8A2BE2)](https://neilkichler.github.io/cuinterval)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16272121.svg)](https://doi.org/10.5281/zenodo.16272121)
[![GitHub License](https://img.shields.io/github/license/neilkichler/cuinterval)](https://github.com/neilkichler/cuinterval/blob/main/LICENSE)

</h1>

CuTangent is a CUDA library for computing forward-mode subgradients, i.e. tangents.

## Supported Operations
`+`
`-`
`*`
`/`
`-x`
`==`
`!=`
`<=>`

`sqr`
`sqrt`
`cbrt`
`abs`
`exp`
`log`
`log2`
`log10`
`pown`
`pow`
`recip`

`sin`
`cos`
`tan`
`asin`
`acos`
`atan`
`atan2`
`sinh`
`cosh`
`tanh`
`asinh`
`acosh` 
`atanh`
`erf`
`erfc`

`max`
`min`
`mid`
`clamp`
`ceil`
`floor`
`remquo`

`isinf`
`isfinite`
`isnan`
`signbit`
`copysign`

## Installation
> Please make sure that you have installed everything mentioned in the section [Build Requirements](#build-requirements).

### System-wide
```bash
git clone https://github.com/neilkichler/cutangent.git
cd cutangent
cmake --preset release
cmake --build build
cmake --install build
```

### CMake Project


#### [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)
```cmake
CPMAddPackage("gh:neilkichler/cutangent@0.0.1")
```

#### [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
```cmake
include(FetchContent)
FetchContent_Declare(
  cutangent
  GIT_REPOSITORY git@github.com:neilkichler/cutangent.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cutangent)
```

In either case, you can link to the library using:
```cmake
target_link_libraries(${PROJECT_NAME} PUBLIC cutangent)
```


> [!IMPORTANT]  
> When using CUDA in a CMake project, make sure that it configures the `CUDA_ARCHITECTURES` property using
> ```cmake
> set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
> ```
> where `native` could be replaced by specific versions, see the [CMake docs](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html) for more information.

## Example
Have a look at the [examples folder](https://github.com/neilkichler/cutangent/tree/main/examples).

## Documentation
The documentation is available [here](https://neilkichler.github.io/cutangent).

## Build

### Build Requirements
We require C++20, CMake v3.25.2+, Ninja, and recent C++ and CUDA compilers.

#### Ubuntu
```bash
apt install cmake gcc ninja-build
```
#### Cluster
```bash
module load CMake CUDA GCC Ninja
```

### Build and run tests
#### Using Workflows
```bash
cmake --workflow --preset dev
```
#### Using Presets
```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```
#### Using regular CMake
```bash
cmake -S . -B build -GNinja
cmake --build build
./build/tests/tests
```
