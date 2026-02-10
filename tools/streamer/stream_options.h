#pragma once

#include <string>

struct stream_options final
{
  std::string send_ip{ "127.0.0.1" };

  int send_port{ 9850 };

  int video_device{ 0 };

  int frame_width{ 1280 };

  int frame_height{ 720 };

  bool benchmark{ false };

  [[nodiscard]] auto parse(int argc, char** argv) -> bool;
};
