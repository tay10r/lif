#include "options.h"

#include <cstdint>

#include <iostream>

#include <cxxopts.hpp>

auto
options::parse(int argc, char** argv) -> bool
{
  cxxopts::Options options("linket_streamer", "A program for streaming video using the linket codec.");

  options.add_options()(
    "a,address", "The IPv4 address to send the tiles to.", cxxopts::value<std::string>()->default_value(send_ip))     //
    ("p,port", "The UDP port to send the tiles to.", cxxopts::value<int>()->default_value(std::to_string(send_port))) //
    ("d,device",
     "The index of the VLC video device to open.",
     cxxopts::value<int>()->default_value(std::to_string(video_device))) //
    ("w,width",
     "The frame width to stream.",
     cxxopts::value<int>()->default_value(std::to_string(frame_width))) //
    ("h,height",
     "The frame height to stream.",
     cxxopts::value<int>()->default_value(std::to_string(frame_height)))                                           //
    ("b,benchmark", "Whether or not to display performance data.", cxxopts::value<bool>()->implicit_value("true")) //
    ("help", "Print this help.", cxxopts::value<bool>()->implicit_value("true"));

  const auto result = options.parse(argc, argv);
  if (result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return false;
  }

  send_ip = result["address"].as<std::string>();
  send_port = result["port"].as<int>();
  video_device = result["device"].as<int>();
  frame_width = result["width"].as<int>();
  frame_height = result["height"].as<int>();
  benchmark = result["benchmark"].as<bool>();

  return true;
}
