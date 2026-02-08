#include "linket_encode.h"

#include "linket_config.h"

void
linket_encoder_forward(const float* input, float* output);

static void
pack_latent(const float* in, unsigned char* out)
{
  float abs_max = 0.0F;

  for (int k = 0; k < LINKET_LATENT_DIM; k++) {
    const float x = in[k];
    const float y = x < 0.0F ? -x : x;
    abs_max = (y > abs_max) ? y : abs_max;
  }

  *((float*)out) = abs_max;

  const float scale = 1.0F / (LINKET_EPSILON + abs_max);

  for (int k = 0; k < LINKET_LATENT_DIM; k++) {
    const float x = ((in[k] * scale) + 1.0F) * 0.5F;
    int v = (int)(x * 255);
    v = (v > 255) ? 255 : v;
    v = (v < 0) ? 0 : v;
    out[k + sizeof(float)] = (unsigned char)v;
  }
}

void
linket_encode(const unsigned char* rgb, const int w, const int h, void* user_data, linket_tile_encode_callback cb)
{
  const int x_tiles = w / LINKET_TILE_SIZE;
  const int y_tiles = h / LINKET_TILE_SIZE;

  const int num_tiles = x_tiles * y_tiles;

  unsigned char latent_bits[LINKET_BYTES_PER_LATENT * LINKET_BLOCKS_PER_TILE];

  for (int i = 0; i < num_tiles; i++) {

    const int x_tile = i % x_tiles;
    const int y_tile = i / x_tiles;

#ifndef LINKET_OPENMP_DISABLED
#pragma omp parallel for
#endif

    for (int j = 0; j < LINKET_BLOCKS_PER_TILE; j++) {

      const int x_block = j % (LINKET_TILE_SIZE / LINKET_BLOCK_SIZE);
      const int y_block = j / (LINKET_TILE_SIZE / LINKET_BLOCK_SIZE);

      const int x_offset = x_tile * LINKET_TILE_SIZE + x_block * LINKET_BLOCK_SIZE;
      const int y_offset = y_tile * LINKET_TILE_SIZE + y_block * LINKET_BLOCK_SIZE;

      const int num_pixels = LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE;

      float net_input[LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE * 3];

      float net_output[LINKET_LATENT_DIM];

      for (int k = 0; k < num_pixels; k++) {

        const int x = k % LINKET_BLOCK_SIZE;
        const int y = k / LINKET_BLOCK_SIZE;

        const unsigned char* pixel = &rgb[((y + y_offset) * w + (x + x_offset)) * 3];

        net_input[0 * LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE + y * LINKET_BLOCK_SIZE + x] =
          ((float)pixel[0]) * (1.0F / 255.0F);
        net_input[1 * LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE + y * LINKET_BLOCK_SIZE + x] =
          ((float)pixel[1]) * (1.0F / 255.0F);
        net_input[2 * LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE + y * LINKET_BLOCK_SIZE + x] =
          ((float)pixel[2]) * (1.0F / 255.0F);
      }

      linket_encoder_forward(net_input, net_output);

      pack_latent(net_output, latent_bits + j * LINKET_BYTES_PER_LATENT);
    }

    cb(user_data, /*x=*/x_tile * LINKET_TILE_SIZE, /*y=*/y_tile * LINKET_TILE_SIZE, latent_bits);
  }
}
