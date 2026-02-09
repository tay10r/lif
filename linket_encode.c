#include "linket_encode.h"

#include "linket_config.h"

void
linket_encoder_forward(const float* input, float* output);

static void
quantize_tile(const float* latent_data, unsigned char* out)
{
  float min_v = latent_data[0];
  float max_v = min_v;

  const int n = LINKET_LATENT_DIM * LINKET_BLOCKS_PER_TILE;

  for (int k = 1; k < n; k++) {
    const float x = latent_data[k];
    min_v = (x < min_v) ? x : min_v;
    max_v = (x > max_v) ? x : max_v;
  }

  ((float*)out)[0] = min_v;
  ((float*)out)[1] = max_v;

  const float scale = 1.0F / (LINKET_EPSILON + (max_v - min_v));

  for (int k = 0; k < n; k++) {
    const float x = ((latent_data[k] - min_v) * scale);
    int v = (int)(x * 255);
    v = (v > 255) ? 255 : v;
    v = (v < 0) ? 0 : v;
    out[k + sizeof(float) * 2] = (unsigned char)v;
  }
}

void
linket_encode(const unsigned char* rgb, const int w, const int h, void* user_data, linket_tile_encode_callback cb)
{
  const int x_tiles = w / LINKET_TILE_SIZE;
  const int y_tiles = h / LINKET_TILE_SIZE;

  const int num_tiles = x_tiles * y_tiles;

  float latent_data[LINKET_LATENT_DIM * LINKET_BLOCKS_PER_TILE];

  unsigned char tile_bits[8 + LINKET_LATENT_DIM * LINKET_BLOCKS_PER_TILE];

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

        const float s = 1.0F / 255.0F;
        net_input[0 * LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE + y * LINKET_BLOCK_SIZE + x] = ((float)pixel[0]) * s;
        net_input[1 * LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE + y * LINKET_BLOCK_SIZE + x] = ((float)pixel[1]) * s;
        net_input[2 * LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE + y * LINKET_BLOCK_SIZE + x] = ((float)pixel[2]) * s;
      }

      linket_encoder_forward(net_input, &latent_data[j * LINKET_LATENT_DIM]);
    }

    quantize_tile(latent_data, tile_bits);

    cb(user_data, /*x=*/x_tile * LINKET_TILE_SIZE, /*y=*/y_tile * LINKET_TILE_SIZE, tile_bits);
  }
}
