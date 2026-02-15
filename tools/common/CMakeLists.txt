cmake_minimum_required(VERSION 3.20)

project(nice_tools_common)

add_library(stb
  stb_image.h
  stb_image.c
  stb_image_write.h
  stb_image_write.c
)

target_include_directories(stb PUBLIC .)
