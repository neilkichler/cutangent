include(FetchContent)
FetchContent_Declare(
  cumccormick
  GIT_REPOSITORY https://github.com/neilkichler/cumccormick.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cumccormick)

find_package(CUDAToolkit REQUIRED)

add_executable(mc mc.cu)

# target_link_libraries(mc PRIVATE CUDA::cuda_driver)
target_link_libraries(mc PRIVATE CUDA::cudart)
target_link_libraries(mc PRIVATE CUDA::curand)

target_link_libraries(mc PUBLIC cumccormick)
target_compile_features(mc PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(mc PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(mc PRIVATE ${PROJECT_NAME})
