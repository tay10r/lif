cmake_minimum_required(VERSION 3.20)

project(linket)

option(LINKET_TOOLS   "Whether or not to build the tools." OFF)
option(LINKET_ENCODER "Whether to build the encoder." ON)
option(LINKET_DECODER "Whether to build the decoder." ON)
option(LINKET_EVAL    "Whether to build the evaluation program." OFF)

find_package(OpenMP REQUIRED COMPONENTS C)

set(compiler_flags)

if(CMAKE_COMPILER_IS_GNUCC)
  set(compiler_flags -ffast-math -march=native -mtune=native)
endif()

if(LINKET_ENCODER)

  add_library(linket_encoder
    linket_encode.h
    linket_encode.c
    linket_encoder_forward.c
  )

  target_include_directories(linket_encoder PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  target_compile_options(linket_encoder PRIVATE ${compiler_flags})

  target_link_libraries(linket_encoder PUBLIC OpenMP::OpenMP_C)

  add_library(linket::encoder ALIAS linket_encoder)

endif()

if(LINKET_DECODER)

  add_library(linket_decoder
    linket_decode.h
    linket_decode.c
    linket_decoder_forward.c
  )

  target_include_directories(linket_decoder PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  target_compile_options(linket_decoder PRIVATE ${compiler_flags})

  target_link_libraries(linket_decoder PUBLIC OpenMP::OpenMP_C)

  add_library(linket::decoder ALIAS linket_decoder)

endif()

if(LINKET_TOOLS)


endif()

if(LINKET_EVAL)

  add_executable(linket_eval
    eval/main.cpp
    eval/algo.h
    eval/algo.cpp
    eval/linket_algo.cpp
    eval/timer.h
    eval/timer.cpp
    eval/deps/stb_image.c
    eval/deps/stb_image.h
    eval/deps/stb_image_write.h
    eval/deps/stb_image_write.c
  )

  target_link_libraries(linket_eval
    PRIVATE
      linket::encoder
      linket::decoder
  )

  target_compile_definitions(linket_eval
    PRIVATE
      "PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
  )

endif()
