#include "lif_encode.h"

#include "lif_config.h"

void
lif_encoder_forward(const float* input, float* output);

void
lif_encode(const unsigned char* rgb, const int w, const int h, void* user_data, lif_tile_encode_callback cb)
{
  const int x_tiles = w / LIF_TILE_SIZE;
  const int y_tiles = h / LIF_TILE_SIZE;

  const int num_tiles = x_tiles * y_tiles;

  float net_input[LIF_BLOCK_SIZE * LIF_BLOCK_SIZE * 3];

  float net_output[LIF_LATENT_DIM];

  unsigned char latent_bits[LIF_LATENT_DIM * LIF_BLOCKS_PER_TILE];

  for (int i = 0; i < num_tiles; i++) {

    const int x_tile = i % x_tiles;
    const int y_tile = i / x_tiles;

#ifndef LIF_OPENMP_DISABLED
#pragma omp parallel for
#endif

    for (int j = 0; j < LIF_BLOCKS_PER_TILE; j++) {

      const int x_block = j % (LIF_TILE_SIZE / LIF_BLOCK_SIZE);
      const int y_block = j / (LIF_TILE_SIZE / LIF_BLOCK_SIZE);

      const int x_offset = x_tile * LIF_TILE_SIZE + x_block * LIF_BLOCK_SIZE;
      const int y_offset = y_tile * LIF_TILE_SIZE + y_block * LIF_BLOCK_SIZE;

      const int num_pixels = LIF_BLOCK_SIZE * LIF_BLOCK_SIZE;

      for (int k = 0; k < num_pixels; k++) {

        const int x = k % LIF_BLOCK_SIZE;
        const int y = k / LIF_BLOCK_SIZE;

        const unsigned char* pixel = &rgb[((y + y_offset) * w + (x + x_offset)) * 3];

        net_input[0 * LIF_BLOCK_SIZE * LIF_BLOCK_SIZE + y * LIF_BLOCK_SIZE + x] = ((float)pixel[0]) * (1.0F / 255.0F);
        net_input[1 * LIF_BLOCK_SIZE * LIF_BLOCK_SIZE + y * LIF_BLOCK_SIZE + x] = ((float)pixel[1]) * (1.0F / 255.0F);
        net_input[2 * LIF_BLOCK_SIZE * LIF_BLOCK_SIZE + y * LIF_BLOCK_SIZE + x] = ((float)pixel[2]) * (1.0F / 255.0F);
      }

      lif_encoder_forward(net_input, net_output);

      for (int k = 0; k < LIF_LATENT_DIM; k++) {
        int v = (int)(net_output[k] * 255);
        v = (v > 255) ? 255 : v;
        v = (v < 0) ? 0 : v;
        latent_bits[j * LIF_LATENT_DIM + k] = (unsigned char)v;
      }
    }

    cb(user_data, /*x=*/x_tile * LIF_TILE_SIZE, /*y=*/y_tile * LIF_TILE_SIZE, latent_bits);
  }
}
