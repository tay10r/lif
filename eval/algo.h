#pragma once

#include <memory>
#include <vector>

#include <stddef.h>
#include <stdint.h>

/// Used for representing a single image compression algorithm.
class algo
{
public:
  [[nodiscard]] static auto create_lif_algo() -> std::unique_ptr<algo>;

  /// The result of the algorithm.
  struct result final
  {
    /// The RGB values after being compressed and decompressed.
    std::vector<uint8_t> rgb;

    /// The number of bytes of the image after compression.
    size_t compressed_size{};
  };

  virtual ~algo();

  [[nodiscard]] virtual auto run(const uint8_t* rgb, int w, int h) -> result = 0;
};
