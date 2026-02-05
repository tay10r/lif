#include <iostream>

#include <cstdlib>

#include "deps/stb_image.h"
#include "deps/stb_image_write.h"

#include "algo.h"

namespace {

class program final
{
public:
  void eval(const stbi_uc* rgb, const int w, const int h)
  {
    {
      auto al = algo::create_linket_algo();

      run_algo(*al, rgb, w, h, "linket_result.png");
    }
  }

protected:
  void run_algo(algo& al, const stbi_uc* rgb, const int w, const int h, const char* output_filename)
  {
    const auto r = al.run(rgb, w, h);

    stbi_write_png(output_filename, w, h, 3, r.rgb.data(), w * 3);
  }
};

} // namespace

auto
main(const int argc, char** argv) -> int
{
  const char* image_path = (argc > 1) ? argv[1] : PROJECT_SOURCE_DIR "/eval/baboon.png";

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
