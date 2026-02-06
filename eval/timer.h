#pragma once

#include <chrono>

class timer final
{
public:
  using clock = std::chrono::high_resolution_clock;

  using time_point = clock::time_point;

  void start();

  [[nodiscard]] auto elapsed() const -> float;

private:
  time_point start_;
};
