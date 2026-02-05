#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  void lif_decode_tile(int x, int y, const unsigned char* latent, int w, int h, unsigned char* rgb);

#ifdef __cplusplus
} /* extern "C" */
#endif
