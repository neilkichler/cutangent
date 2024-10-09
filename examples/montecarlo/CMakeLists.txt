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


add_executable(mc_heston mc_heston.cu)
target_link_libraries(mc_heston PRIVATE CUDA::cudart)
target_link_libraries(mc_heston PRIVATE CUDA::curand)

target_link_libraries(mc_heston PUBLIC cumccormick)
target_compile_features(mc_heston PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(mc_heston PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(mc_heston PRIVATE ${PROJECT_NAME})
