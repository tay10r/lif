#include <iostream>
#include <vector>

#include <NICE.h>

#include <cstdlib>

#include "deps/stb_image.h"
#include "deps/stb_image_write.h"

#include "timer.h"

#include <assert.h>
#include <string.h>

namespace {

class program final
{
public:
  void eval(const stbi_uc* rgb, const int w, const int h)
  {
    const auto pad_w = ((w + 79) / 80) * 80;
    const auto pad_h = ((h + 79) / 80) * 80;

    std::vector<stbi_uc> padded(pad_w * pad_h * 3, 0);

    for (int y = 0; y < h; y++) {
      memcpy(padded.data() + y * pad_w * 3, rgb + y * w * 3, w * 3);
    }

    eval_padded(padded.data(), pad_w, pad_h);
  }

protected:
  void eval_padded(const stbi_uc* rgb, const int w, const int h)
  {
    assert(w % 80 == 0);
    assert(h % 80 == 0);

    std::vector<stbi_uc> rgb_output(w * h * 3, 0);

    const int pitch = w * 3;

    float compress_time{ 0.0F };

    float decompress_time{ 0.0F };

    auto* engine = NICE_NewEngine();

    timer t;

    for (int y = 0; y < h; y += 80) {

      for (int x = 0; x < w; x += 80) {

        unsigned char bits[12 * 100]{};

        t.start();

        NICE_EncodeTile(engine, rgb + (y * w + x) * 3, pitch, bits);

        compress_time += t.elapsed();

        t.start();

        NICE_DecodeTile(engine, bits, pitch, rgb_output.data() + (y * w + x) * 3);

        decompress_time += t.elapsed();
      }
    }

    NICE_DestroyEngine(engine);

    stbi_write_png("result.png", w, h, 3, rgb_output.data(), pitch);

    std::cout << "compress time: " << compress_time << std::endl;
    std::cout << "decompress time: " << decompress_time << std::endl;
  }
};

} // namespace

auto
main(const int argc, char** argv) -> int
{
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <image-path> <encoder-weights> <decoder-weights>" << std::endl;
    return EXIT_FAILURE;
  }

  const char* image_path = argv[1];

  int w = 0;
  int h = 0;
  auto* rgb = stbi_load(image_path, &w, &h, nullptr, 3);
  if (!rgb) {
    std::cerr << "failed to open \"" << image_path << '\"' << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "opened \"" << image_path << '"' << std::endl;

  program prg;

  prg.eval(rgb, w, h);

  stbi_image_free(rgb);

  return EXIT_SUCCESS;
}
