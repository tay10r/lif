cmake_minimum_required(VERSION 3.20)

project(preprocess)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(stb_image
  stb_image.h
  stb_image.c
)

add_executable(preprocess
  main.cpp
)

target_link_libraries(preprocess
  PUBLIC
    OpenMP::OpenMP_CXX
    stb_image
)
