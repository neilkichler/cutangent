include(FetchContent)
FetchContent_Declare(
  cumccormick
  GIT_REPOSITORY https://github.com/neilkichler/cumccormick.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cumccormick)

add_example(mccormick)
target_link_libraries(mccormick PUBLIC cumccormick)
