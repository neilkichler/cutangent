include(FetchContent)

# Testing library
FetchContent_Declare(
  ut
  GIT_REPOSITORY https://github.com/boost-ext/ut.git
  GIT_TAG v2.1.1
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ut)

# Tests need to be added as executables first
add_executable(tests 
    tests.cpp
    tests_against_finite_differences.cu
)

# Use C++20 in the tests
target_compile_features(tests PRIVATE cxx_std_20 cuda_std_20)

# Specify the CUDA architecture to compile to
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES native)

# Enable separate compilation
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link with main library and testing library
target_link_libraries(tests PRIVATE cutangent ut warnings)

# Better constexpr support
target_compile_options(tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")

# Better lambda support
target_compile_options(tests PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>")

# Find and link with CUDA headers
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tests PRIVATE CUDA::cuda_driver)
target_link_libraries(tests PRIVATE CUDA::cudart)

# Register tests for use in ctest
add_test(NAME testcutangent COMMAND tests)

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
if(ENABLE_TEST_COVERAGE)
    target_compile_options(tests PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
    target_link_options(tests PUBLIC -fprofile-arcs -ftest-coverage)
endif()
