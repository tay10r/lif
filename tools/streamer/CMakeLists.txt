cmake_minimum_required(VERSION 3.20)

project(nice_streamer)

find_package(spdlog REQUIRED)
find_package(cxxopts REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(UV REQUIRED libuv)

if(NOT TARGET NICE::NICE)
  add_subdirectory(../.. core)
endif()

add_executable(stream
  main.cpp
  options.h
  options.cpp
  camera.h
  camera.cpp
)

target_include_directories(stream
  PRIVATE
    ${UV_INCLUDE_DIRS}
)

target_link_libraries(stream
  PRIVATE
    NICE::NICE
    spdlog::spdlog
    cxxopts::cxxopts
    ${UV_LIBRARIES}
)
