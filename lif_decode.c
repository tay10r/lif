#include "lif_decode.h"

#include "lif_config.h"

void
lif_decoder_forward(const float* input, float* output);

void
lif_decode_tile(const int x, const int y, const unsigned char* latent, const int w, const int h, unsigned char* rgb)
{
  const int blocks_per_row = (LIF_TILE_SIZE / LIF_BLOCK_SIZE);
  const int num_pixels = LIF_BLOCK_SIZE * LIF_BLOCK_SIZE;

#ifndef LIF_OPENMP_DISABLED
#pragma omp parallel for
#endif

  for (int j = 0; j < LIF_BLOCKS_PER_TILE; j++) {

    float net_input[LIF_LATENT_DIM];

    float net_output[LIF_BLOCK_SIZE * LIF_BLOCK_SIZE * 3];

    const int x_block = j % blocks_per_row;
    const int y_block = j / blocks_per_row;

    const int x_offset = x + x_block * LIF_BLOCK_SIZE;
    const int y_offset = y + y_block * LIF_BLOCK_SIZE;

    for (int k = 0; k < LIF_LATENT_DIM; k++) {
      net_input[k] = ((float)latent[j * LIF_LATENT_DIM + k]) * (1.0F / 255.0F);
    }

    lif_decoder_forward(net_input, net_output);

    for (int p = 0; p < num_pixels; p++) {

      const int bx = p % LIF_BLOCK_SIZE;
      const int by = p / LIF_BLOCK_SIZE;

      const int ix = x_offset + bx;
      const int iy = y_offset + by;

      if ((unsigned)ix >= (unsigned)w || (unsigned)iy >= (unsigned)h) {
        continue;
      }

      const int out_idx = by * LIF_BLOCK_SIZE + bx;

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
