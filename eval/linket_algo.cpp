#include "algo.h"

#include <linket_config.h>
#include <linket_decode.h>
#include <linket_encode.h>

#include <string.h>

namespace {

class lif_algo final : public algo
{
public:
  [[nodiscard]] auto run(const uint8_t* rgb, const int w, const int h) -> result override
  {
    width_ = w;

    height_ = h;

    const int x_tiles = w / LIF_TILE_SIZE;
    const int y_tiles = h / LIF_TILE_SIZE;

    const size_t num_tiles = (size_t)x_tiles * (size_t)y_tiles;

    latent_buffer_.resize(num_tiles * LIF_BLOCKS_PER_TILE * LIF_LATENT_DIM);

    latent_buffer_offset_ = 0;

    result_.rgb.resize(w * h * 3);

    linket_encode(rgb, w, h, this, on_tile);

    result_.compressed_size = latent_buffer_offset_;

    return std::move(result_);
  }

protected:
  static void on_tile(void* self_ptr, const int x, const int y, const uint8_t* latent)
  {
    auto* self = static_cast<lif_algo*>(self_ptr);

    const size_t s = LIF_LATENT_DIM * LIF_BLOCKS_PER_TILE;

    memcpy(self->latent_buffer_.data() + self->latent_buffer_offset_, latent, s);

    self->latent_buffer_offset_ += s;

    linket_decode_tile(x, y, latent, self->width_, self->height_, self->result_.rgb.data());
  }

private:
  int width_{};

  int height_{};

  result result_;

  std::vector<uint8_t> latent_buffer_;

  size_t latent_buffer_offset_{};
};

} // namespace

auto
algo::create_linket_algo() -> std::unique_ptr<algo>
{
  return std::make_unique<lif_algo>();
}
