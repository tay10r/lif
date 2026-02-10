#include "timer.h"

void
timer::start()
{
  start_ = clock::now();
}

auto
timer::elapsed() const -> float
{
  auto t = clock::now();
  const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t - start_).count();
  return static_cast<float>(dt) * 1.0e-6F;
}
