cmake_minimum_required(VERSION 3.20)

project(NICE)

option(NICE_EVAL     "Whether to build the eval program."         OFF)
option(NICE_PRETRAIN "Whether to build the tool for pretraining." OFF)

find_package(OpenMP REQUIRED COMPONENTS C)

set(compiler_flags)

if(CMAKE_COMPILER_IS_GNUCC)
  set(compiler_flags -ffast-math -march=native -mtune=native)
endif()

add_library(NICE
  NICE.h
  NICE.c
  NICE_Kernels.h
  NICE_Kernels.c
  NICE_Weights.h
  NICE_Weights.c
)

target_compile_features(NICE PUBLIC c_std_11)

target_compile_options(NICE PUBLIC ${compiler_flags})

target_include_directories(NICE PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(NICE PUBLIC OpenMP::OpenMP_C)

add_library(NICE::NICE ALIAS NICE)

if(NICE_EVAL)
  add_subdirectory(tools/eval)
endif()

if(NICE_PRETRAIN)
  add_subdirectory(tools/pretrain)
endif()

if(NICE_STREAM)
endif(NICE_STREAM)
