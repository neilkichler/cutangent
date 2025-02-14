include(FetchContent)
FetchContent_Declare(
  cumccormick
  GIT_REPOSITORY https://github.com/neilkichler/cumccormick.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cumccormick)

add_executable(blackscholes blackscholes.cu)
target_link_libraries(blackscholes PUBLIC cumccormick)
target_compile_features(blackscholes PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(blackscholes PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(blackscholes PRIVATE ${PROJECT_NAME} warnings)
