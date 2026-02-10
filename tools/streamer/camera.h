#pragma once

#include <memory>
#include <utility>

class camera
{
public:
  [[nodiscard]] static auto create(int device_index, int w, int h) -> std::unique_ptr<camera>;

  virtual ~camera() = default;

  [[nodiscard]] virtual auto wait_frame() -> unsigned char* = 0;

  [[nodiscard]] virtual auto get_frame_size() const -> std::pair<int, int> = 0;
};
