cmake_minimum_required(VERSION 3.20)

project(lif)

option(LIF_TOOLS   "Whether or not to build the tools." OFF)
option(LIF_ENCODER "Whether to build the encoder." ON)
option(LIF_DECODER "Whether to build the decoder." ON)
option(LIF_EVAL    "Whether to build the evaluation program." OFF)


if(LIF_ENCODER)

  add_library(lif_encoder
    lif_encode.h
    lif_encode.c
    lif_encoder_forward.c
  )

  target_include_directories(lif_encoder PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  add_library(lif::encoder ALIAS lif_encoder)

endif()

if(LIF_DECODER)

  add_library(lif_decoder
    lif_decode.h
    lif_decode.c
    lif_decoder_forward.c
  )

  target_include_directories(lif_decoder PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

  add_library(lif::decoder ALIAS lif_decoder)

endif()

if(LIF_TOOLS)


endif()

if(LIF_EVAL)

  add_executable(lif_eval
    eval/main.cpp
    eval/algo.h
    eval/algo.cpp
    eval/lif_algo.cpp
    eval/deps/stb_image.c
    eval/deps/stb_image.h
    eval/deps/stb_image_write.h
    eval/deps/stb_image_write.c
  )

  target_link_libraries(lif_eval
    PRIVATE
      lif::encoder
      lif::decoder
  )

  target_compile_definitions(lif_eval
    PRIVATE
      "PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\""
  )

endif()
