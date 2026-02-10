cmake_minimum_required(VERSION 3.20)

project(linket_streamer)

if(NOT TARGET linket::core)
  add_subdirectory(../.. core)
endif()

include(cmake/glow.cmake)

add_executable(linket_viewer WIN32
  main.cpp
)

target_include_directories(linket_viewer
  PRIVATE
    ${UV_INCLUDE_DIRS}
)

target_link_libraries(linket_viewer
  PRIVATE
    glow::glow
    glow::main
    linket::core
    ${UV_LIBRARIES}
)
