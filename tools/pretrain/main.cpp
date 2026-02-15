#include <NICE.h>

#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>

#include <stb_image.h>
#include <stb_image_write.h>

namespace {

namespace fs = std::filesystem;

class Image final
{
public:
  Image(const fs::path& path)
  {
    image_ = stbi_load(path.string().c_str(), &width_, &height_, nullptr, 3);
    if (!image_) {
      std::ostringstream what;
      what << "failed to open " << path;
      throw std::runtime_error(what.str());
    }
  }

  Image(Image&& other) noexcept
    : image_(other.image_)
    , width_(other.width_)
    , height_(other.height_)
  {
    other.image_ = nullptr;
  }

  Image(const Image&) = delete;

  ~Image() { stbi_image_free(image_); }

  [[nodiscard]] auto data() const -> const stbi_uc* { return image_; }

  [[nodiscard]] auto width() const -> int { return width_; }

  [[nodiscard]] auto height() const -> int { return height_; }

private:
  stbi_uc* image_{};

  int width_{};

  int height_{};
};

class Program final
{
public:
  ~Program() { NICE_DestroyEngine(engine_); }

  [[nodiscard]] auto run(const int num_epochs = 128, const float lr = 1.0e-3F) -> bool
  {
    if (!load_data()) {
      return false;
    }

    for (int epoch = 0; epoch < num_epochs; epoch++) {

      train_epoch(lr);

      const auto is_last = (epoch + 1) == num_epochs;

      const auto loss_avg = val_epoch(/*save_images=*/is_last);

      std::printf("[%d]: %.06f\n", epoch, loss_avg);
    }

    return true;
  }

protected:
  void train_epoch(const float lr, const int num_repeats = 8192)
  {
    for (auto iteration = 0; iteration < num_repeats; iteration++) {

      for (const auto& img : train_images_) {

        std::uniform_int_distribution<int> x_dist(0, img.width() - NICE_BLOCK_SIZE);

        std::uniform_int_distribution<int> y_dist(0, img.height() - NICE_BLOCK_SIZE);

        const auto x = x_dist(rng_);
        const auto y = y_dist(rng_);

        const auto* rgb = img.data() + (y * img.width() + x) * 3;

        NICE_TrainBlock(engine_, rgb, img.width() * 3, lr);
      }
    }
  }

  [[nodiscard]] auto mse(const unsigned char* rgb_original, const unsigned char* rgb_out, const int pitch) const
    -> float
  {
    auto error{ 0.0F };

    for (auto y = 0; y < NICE_BLOCK_SIZE; y++) {

      for (auto x = 0; x < NICE_BLOCK_SIZE; x++) {

        const auto* in = rgb_original + y * pitch + x * 3;
        const auto* out = rgb_out + y * pitch + x * 3;

        constexpr auto s{ 1.0F / 255.0F };
        const auto d0 = (static_cast<float>(in[0]) - static_cast<float>(out[0])) * s;
        const auto d1 = (static_cast<float>(in[1]) - static_cast<float>(out[1])) * s;
        const auto d2 = (static_cast<float>(in[2]) - static_cast<float>(out[2])) * s;

        error += d0 * d0 + d1 * d1 + d2 * d2;
      }
    }

    error *= (1.0F / (NICE_BLOCK_SIZE * NICE_BLOCK_SIZE * 3));

    return error;
  }

  [[nodiscard]] auto val_epoch(const bool save_images) const -> float
  {
    auto loss_sum{ 0.0F };

    std::vector<unsigned char> result_buffer;

    auto counter{ 0 };

    for (const auto& img : val_images_) {

      auto img_loss = 0.0F;

      result_buffer.resize(img.width() * img.height() * 3);

      for (int y = 0; y < img.height(); y += NICE_BLOCK_SIZE) {

        for (int x = 0; x < img.width(); x += NICE_BLOCK_SIZE) {

          const auto* ptr = img.data() + (y * img.width() + x) * 3;

          unsigned char bits[4]{};

          NICE_Encode(engine_, ptr, img.width() * 3, bits);

          auto* dst = result_buffer.data() + (y * img.width() + x) * 3;

          NICE_Decode(engine_, bits, /*pitch=*/img.width() * 3, dst);

          img_loss += mse(ptr, dst, img.width() * 3);
        }
      }

      counter++;

      if (save_images) {
        std::ostringstream path;
        path << std::setw(2) << std::setfill('0') << counter << ".png";
        stbi_write_png(path.str().c_str(), img.width(), img.height(), 3, result_buffer.data(), img.width() * 3);
      }

      const auto xb = img.width() / NICE_BLOCK_SIZE;
      const auto yb = img.height() / NICE_BLOCK_SIZE;
      loss_sum += img_loss * (1.0F / static_cast<float>(xb * yb));
    }

    const auto avg_loss = loss_sum * (1.0F / static_cast<float>(val_images_.size()));

    return avg_loss;
  }

  [[nodiscard]] static auto load_images(const fs::path& dir) -> std::vector<Image>
  {
    std::vector<Image> images;

    for (const auto& entry : fs::directory_iterator(dir)) {
      images.emplace_back(entry.path().string().c_str());
    }

    return images;
  }

  [[nodiscard]] auto load_data() -> bool
  {
    const auto root = std::filesystem::path(DATA_ROOT);
    train_images_ = load_images(root / "train");
    val_images_ = load_images(root / "val");
    return true;
  }

private:
  NICE_Engine* engine_{ NICE_NewEngine() };

  std::mt19937 rng_{ 0 };

  std::vector<Image> train_images_;

  std::vector<Image> val_images_;
};

} // namespace

int
main()
{
  Program program;

  return program.run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
