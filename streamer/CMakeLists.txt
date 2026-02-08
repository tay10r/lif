cmake_minimum_required(VERSION 3.20)

project(linket_streamer)

include(cmake/glow.cmake)

#========#
# Viewer #
#========#

add_executable(linket_viewer
  viewer.cpp
)

target_include_directories(linket_viewer
  PRIVATE
    ${UV_INCLUDE_DIRS}
)

target_link_libraries(linket_viewer
  PRIVATE
    glow::glow
    glow::main
    linket::decoder
    ${UV_LIBRARIES}
)

#==========#
# Streamer #
#==========#

find_package(PkgConfig REQUIRED)

pkg_check_modules(UV REQUIRED libuv)

if(NOT TARGET linket::encoder)
  add_subdirectory(.. linket)
endif()

add_executable(linket_stream
  main.cpp
  camera.h
  camera.cpp
)

target_include_directories(linket_stream
  PRIVATE
    ${UV_INCLUDE_DIRS}
)

target_link_libraries(linket_stream
  PRIVATE
    linket::encoder
    ${UV_LIBRARIES}
)
