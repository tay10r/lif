#include "linket_decode.h"

#include "linket_config.h"

void
linket_decoder_forward(const float* input, float* output);

void
linket_decode_tile(const int x, const int y, const unsigned char* latent, const int w, const int h, unsigned char* rgb)
{
  const int blocks_per_row = (LINKET_TILE_SIZE / LINKET_BLOCK_SIZE);

  const int num_pixels = LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE;

#ifndef LINKET_OPENMP_DISABLED
#pragma omp parallel for
#endif

  for (int j = 0; j < LINKET_BLOCKS_PER_TILE; j++) {

    float net_input[LINKET_LATENT_DIM];

    float net_output[LINKET_BLOCK_SIZE * LINKET_BLOCK_SIZE * 3];

    const int x_block = j % blocks_per_row;
    const int y_block = j / blocks_per_row;

    const int x_offset = x + x_block * LINKET_BLOCK_SIZE;
    const int y_offset = y + y_block * LINKET_BLOCK_SIZE;

    const float scale = *((const float*)(latent + j * LINKET_BYTES_PER_LATENT));

    for (int k = 0; k < LINKET_LATENT_DIM; k++) {

      const float x = ((float)latent[j * LINKET_BYTES_PER_LATENT + k + sizeof(float)]) * (1.0F / 255.0F);
      const float y = (x * 2.0F - 1.0F) * scale;
      net_input[k] = y;
    }

    linket_decoder_forward(net_input, net_output);

    for (int p = 0; p < num_pixels; p++) {

      const int bx = p % LINKET_BLOCK_SIZE;
      const int by = p / LINKET_BLOCK_SIZE;

      const int ix = x_offset + bx;
      const int iy = y_offset + by;

      if ((unsigned)ix >= (unsigned)w || (unsigned)iy >= (unsigned)h) {
        continue;
      }

      const int out_idx = by * LINKET_BLOCK_SIZE + bx;

      const float r_f = net_output[0 * num_pixels + out_idx];
      const float g_f = net_output[1 * num_pixels + out_idx];
      const float b_f = net_output[2 * num_pixels + out_idx];

      int r = (int)(r_f * 255.0F);
      int g = (int)(g_f * 255.0F);
      int b = (int)(b_f * 255.0F);

      r = (r > 255) ? 255 : r;
      r = (r < 0) ? 0 : r;
      g = (g > 255) ? 255 : g;
      g = (g < 0) ? 0 : g;
      b = (b > 255) ? 255 : b;
      b = (b < 0) ? 0 : b;

      unsigned char* dst = &rgb[(iy * w + ix) * 3];
      dst[0] = (unsigned char)r;
      dst[1] = (unsigned char)g;
      dst[2] = (unsigned char)b;
    }
  }
}
