include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY https://github.com/neilkichler/cuinterval.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cuinterval)

add_executable(natural natural.cu)
target_link_libraries(natural PUBLIC cuinterval)
target_compile_features(natural PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(natural PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(natural PRIVATE ${PROJECT_NAME})

add_executable(centered centered.cu)
target_link_libraries(centered PUBLIC cuinterval)
target_compile_features(centered PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(centered PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(centered PRIVATE ${PROJECT_NAME})
