include(FetchContent)
FetchContent_Declare(
  cuinterval
  GIT_REPOSITORY https://github.com/neilkichler/cuinterval.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cuinterval)

add_example(natural)
target_link_libraries(natural PUBLIC cuinterval)

add_example(centered)
target_link_libraries(centered PUBLIC cuinterval)
