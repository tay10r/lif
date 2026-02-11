cmake_minimum_required(VERSION 3.20)

project(nice_streamer)

find_package(PkgConfig REQUIRED)

pkg_check_modules(UV REQUIRED libuv)

if(NOT TARGET NICE::NICE)
  add_subdirectory(../.. core)
endif()

include(cmake/glow.cmake)

add_executable(viewer WIN32
  main.cpp
)

target_include_directories(viewer
  PRIVATE
    ${UV_INCLUDE_DIRS}
)

target_link_libraries(viewer
  PRIVATE
    glow::glow
    glow::main
    NICE::NICE
    ${UV_LIBRARIES}
)
