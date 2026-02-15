cmake_minimum_required(VERSION 3.20)

project(nice_pretrain)

if(NOT TARGET stb)
  add_subdirectory(../common common)
endif()

add_executable(pretrain main.cpp)

target_link_libraries(pretrain PRIVATE NICE::NICE stb)

target_compile_definitions(pretrain PRIVATE "DATA_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/../../data\"")
