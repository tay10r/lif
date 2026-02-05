#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "stb_image.h"

using u8 = std::uint8_t;
namespace fs = std::filesystem;

[[nodiscard]] auto
is_jpeg(const fs::path& p) -> bool
{
  auto e = p.extension().string();
  for (auto& c : e)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e == ".jpg" || e == ".jpeg";
}

[[nodiscard]] auto
load_rgb(const fs::path& p) -> std::optional<std::tuple<std::vector<u8>, int, int>>
{
  int w = 0, h = 0, n = 0;
  auto* d = stbi_load(p.string().c_str(), &w, &h, &n, 3);
  if (!d)
    return std::nullopt;
  std::vector<u8> img(static_cast<std::size_t>(w) * h * 3);
  std::memcpy(img.data(), d, img.size());
  stbi_image_free(d);
  return std::make_tuple(std::move(img), w, h);
}

[[nodiscard]] auto
floor8(int v) -> int
{
  return (v / 8) * 8;
}

int
main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <input_dir> <output.bin>\n";
    return 1;
  }

  fs::path input = argv[1];

  fs::path output = argv[2];

  std::vector<fs::path> files;

  for (const auto& e : fs::recursive_directory_iterator(input)) {
    if (e.is_regular_file() && is_jpeg(e.path())) {
      files.push_back(e.path());
    }
  }

  std::sort(files.begin(), files.end());

  constexpr size_t samples_per_image{ 128 };

  constexpr size_t dim_size{ 8 };

  constexpr size_t block_size{ 3 * dim_size * dim_size };

  const size_t num_images = files.size();

  std::vector<uint8_t> out_buffer(samples_per_image * block_size * num_images);

  std::array<u8, block_size> block{};

#pragma omp parallel for

  for (long i = 0; i < static_cast<long>(num_images); i++) {

    const auto& p = files[i];

    std::seed_seq seed{ i };

    std::mt19937 rng(seed);

    auto loaded = load_rgb(p);
    if (!loaded) {
      continue;
    }

    auto& [img, w, h] = *loaded;
    if ((w < 8) || (h < 8)) {
      continue;
    }

    std::uniform_int_distribution<int> x_dist(0, w - dim_size);
    std::uniform_int_distribution<int> y_dist(0, h - dim_size);

    for (size_t s = 0; s < samples_per_image; s++) {

      const auto x0 = x_dist(rng);
      const auto y0 = y_dist(rng);

      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dim_size; ++y) {
          auto row = static_cast<std::size_t>((y0 + y) * w * 3);
          for (int x = 0; x < dim_size; ++x) {
            block[c * dim_size * dim_size + y * dim_size + x] = img[row + static_cast<std::size_t>((x0 + x) * 3 + c)];
          }
        }
      }

      const auto offset = (i * samples_per_image + s) * block_size;

      std::memcpy(out_buffer.data() + offset, block.data(), block.size());
    }
  }

  std::ofstream out(output, std::ios::binary);
  if (!out) {
    std::cerr << "failed to open output\n";
    return 1;
  }

  out.write(reinterpret_cast<const char*>(out_buffer.data()), out_buffer.size());

  out.flush();

  return 0;
}
