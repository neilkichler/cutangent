add_executable(interval interval.cu)

include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY https://github.com/neilkichler/cuinterval.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cuinterval)

target_link_libraries(interval PUBLIC cuinterval)

target_compile_features(interval PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(interval PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(interval PRIVATE ${PROJECT_NAME})
