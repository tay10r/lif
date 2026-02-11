cmake_minimum_required(VERSION 3.20)

project(NICE_Eval)

add_executable(nice_eval
  main.cpp
  timer.h
  timer.cpp
  deps/stb_image.c
  deps/stb_image.h
  deps/stb_image_write.h
  deps/stb_image_write.c
)

target_link_libraries(nice_eval
  PRIVATE
    NICE::NICE
)

target_compile_definitions(nice_eval
  PRIVATE
    "PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
)
